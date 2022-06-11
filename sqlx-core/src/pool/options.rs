use crate::connection::Connection;
use crate::database::Database;
use crate::error::Error;
use crate::pool::inner::PoolInner;
use crate::pool::Pool;
use futures_core::future::BoxFuture;
use sqlx_rt::spawn;
use std::cmp;
use std::fmt::{self, Debug, Formatter};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Configuration options for [`Pool`][super::Pool].
///
/// ### Callback Functions: Why Do I Need `Box::pin()`?
/// Essentially, because it's impossible to write generic bounds that describe a closure
/// with a higher-ranked lifetime parameter, returning a future with that same lifetime.
///
/// Ideally, you could define it like this:
/// ```rust,ignore
/// async fn takes_foo_callback(f: impl for<'a> Fn(&'a mut Foo) -> impl Future<'a, Output = ()>)
/// ```
///
/// However, the compiler does not allow using `impl Trait` in the return type of an `impl Fn`.
///
/// And if you try to do it like this:
/// ```rust,ignore
/// async fn takes_foo_callback<F, Fut>(f: F)
/// where
///     F: for<'a> Fn(&'a mut Foo) -> Fut,
///     Fut: for<'a> Future<Output = ()> + 'a
/// ```
///
/// There's no way to tell the compiler that those two `'a`s should be the same lifetime.
///
/// It's possible to make this work with a custom trait, but it's fiddly and requires naming
///  the type of the closure parameter.
///
/// Having the closure return `BoxFuture` allows us to work around this, as all the type information
/// fits into a single generic parameter.
///
/// We still need to `Box` the future internally to give it a concrete type to avoid leaking a type
/// parameter everywhere, and `Box` is in the prelude so it doesn't need to be manually imported,
/// so having the closure return `BoxFuture` directly is the path of least resistance from the
/// perspective of both API designer and consumer.
pub struct PoolOptions<DB: Database> {
    pub(crate) test_before_acquire: bool,
    pub(crate) after_connect: Option<
        Box<
            dyn Fn(&mut DB::Connection, PoolConnectionMetadata) -> BoxFuture<'_, Result<(), Error>>
                + 'static
                + Send
                + Sync,
        >,
    >,
    pub(crate) before_acquire: Option<
        Box<
            dyn Fn(
                    &mut DB::Connection,
                    PoolConnectionMetadata,
                ) -> BoxFuture<'_, Result<bool, Error>>
                + 'static
                + Send
                + Sync,
        >,
    >,
    pub(crate) after_release: Option<
        Box<
            dyn Fn(
                    &mut DB::Connection,
                    PoolConnectionMetadata,
                ) -> BoxFuture<'_, Result<bool, Error>>
                + 'static
                + Send
                + Sync,
        >,
    >,
    pub(crate) max_connections: u32,
    pub(crate) acquire_timeout: Duration,
    pub(crate) min_connections: u32,
    pub(crate) max_lifetime: Option<Duration>,
    pub(crate) idle_timeout: Option<Duration>,
    pub(crate) fair: bool,
}

/// Metadata for the connection being processed by a [`PoolOptions`] callback.
#[derive(Debug)] // Don't want to commit to any other trait impls yet.
#[non_exhaustive] // So we can safely add fields in the future.
pub struct PoolConnectionMetadata {
    /// The duration since the connection was first opened.
    ///
    /// For [`after_connect`][Self::after_connect], this is [`Duration::ZERO`].
    pub age: Duration,

    /// The duration that the connection spent in the idle queue.
    ///
    /// Only relevant for [`before_acquire`][Self::before_acquire].
    /// For other callbacks, this is [`Duration::ZERO`].
    pub idle_for: Duration,
}

impl<DB: Database> Default for PoolOptions<DB> {
    fn default() -> Self {
        Self::new()
    }
}

impl<DB: Database> PoolOptions<DB> {
    /// Returns a default "sane" configuration, suitable for testing or light-duty applications.
    ///
    /// Production applications will likely want to at least modify
    /// [`max_connections`][Self::max_connections].
    ///
    /// See the source of this method for the current default values.
    pub fn new() -> Self {
        Self {
            // User-specifiable routines
            after_connect: None,
            before_acquire: None,
            after_release: None,
            test_before_acquire: true,
            // A production application will want to set a higher limit than this.
            max_connections: 10,
            min_connections: 0,
            acquire_timeout: Duration::from_secs(30),
            idle_timeout: Some(Duration::from_secs(10 * 60)),
            max_lifetime: Some(Duration::from_secs(30 * 60)),
            fair: true,
        }
    }

    /// Set the maximum number of connections that this pool should maintain.
    ///
    /// Be mindful of the connection limits for your database as well as other applications
    /// which may want to connect to the same database (or even multiple instances of the same
    /// application in high-availability deployments).
    pub fn max_connections(mut self, max: u32) -> Self {
        self.max_connections = max;
        self
    }

    /// Set the maximum amount of time to spend waiting for a connection in [`Pool::acquire()`].
    ///
    /// If this timeout elapses, [`Pool::acquire`] will return an error.
    pub fn acquire_timeout(mut self, timeout: Duration) -> Self {
        self.acquire_timeout = timeout;
        self
    }

    /// Set the minimum number of connections to maintain at all times.
    ///
    /// When the pool is built, this many connections will be automatically spun up.
    ///
    /// If any connection is reaped by [`max_lifetime`] or [`idle_timeout`] and it brings
    /// the connection count below this amount, a new connection will be opened to replace it.
    ///
    /// [`max_lifetime`]: Self::max_lifetime
    /// [`idle_timeout`]: Self::idle_timeout
    pub fn min_connections(mut self, min: u32) -> Self {
        self.min_connections = min;
        self
    }

    /// Set the maximum lifetime of individual connections.
    ///
    /// Any connection with a lifetime greater than this will be closed.
    ///
    /// When set to `None`, all connections live until either reaped by [`idle_timeout`]
    /// or explicitly disconnected.
    ///
    /// Infinite connections are not recommended due to the unfortunate reality of memory/resource
    /// leaks on the database-side. It is better to retire connections periodically
    /// (even if only once daily) to allow the database the opportunity to clean up data structures
    /// (parse trees, query metadata caches, thread-local storage, etc.) that are associated with a
    /// session.
    ///
    /// [`idle_timeout`]: Self::idle_timeout
    pub fn max_lifetime(mut self, lifetime: impl Into<Option<Duration>>) -> Self {
        self.max_lifetime = lifetime.into();
        self
    }

    /// Set a maximum idle duration for individual connections.
    ///
    /// Any connection with an idle duration longer than this will be closed.
    ///
    /// For usage-based database server billing, this can be a cost saver.
    pub fn idle_timeout(mut self, timeout: impl Into<Option<Duration>>) -> Self {
        self.idle_timeout = timeout.into();
        self
    }

    /// If true, the health of a connection will be verified by a call to [`Connection::ping`]
    /// before returning the connection.
    ///
    /// Defaults to `true`.
    pub fn test_before_acquire(mut self, test: bool) -> Self {
        self.test_before_acquire = test;
        self
    }

    /// If set to `true`, calls to `acquire()` are fair and connections  are issued
    /// in first-come-first-serve order. If `false`, "drive-by" tasks may steal idle connections
    /// ahead of tasks that have been waiting.
    ///
    /// According to `sqlx-bench/benches/pg_pool` this may slightly increase time
    /// to `acquire()` at low pool contention but at very high contention it helps
    /// avoid tasks at the head of the waiter queue getting repeatedly preempted by
    /// these "drive-by" tasks and tasks further back in the queue timing out because
    /// the queue isn't moving.
    ///
    /// Currently only exposed for benchmarking; `fair = true` seems to be the superior option
    /// in most cases.
    #[doc(hidden)]
    pub fn __fair(mut self, fair: bool) -> Self {
        self.fair = fair;
        self
    }

    /// Perform an asynchronous action after connecting to the database.
    ///
    /// If the operation returns with an error then the error is logged, the connection is closed
    /// and a new one is opened in its place and the callback is invoked again.
    ///
    /// This occurs in a backoff loop to avoid high CPU usage and spamming logs during a transient
    /// error condition.
    ///
    /// # Example: Additional Parameters
    /// This callback may be used to set additional configuration parameters
    /// that are not exposed by the database's `ConnectOptions`.
    ///
    /// This example is written for PostgreSQL but can likely be adapted to other databases.
    ///
    /// ```no_run
    /// # async fn f() -> Result<(), Box<dyn std::error::Error>> {
    /// use sqlx::Executor;
    /// use sqlx::postgres::PgPoolOptions;
    ///
    /// let pool = PgPoolOptions::new()
    ///     .after_connect(|conn, _meta| Box::pin(async move {
    ///         // When directly invoking `Executor` methods,
    ///         // it is possible to execute multiple statements with one call.
    ///         conn.execute("SET application_name = 'your_app'; SET search_path = 'my_schema';")
    ///             .await?;
    ///
    ///         Ok(())
    ///     }))    
    ///     .connect("postgres:// …").await?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// For a discussion on why `Box::pin()` is required, see [the type-level docs][Self].
    pub fn after_connect<F>(mut self, callback: F) -> Self
    where
        // We're passing the `PoolConnectionMetadata` here mostly for future-proofing.
        // `age` and `idle_for` are obviously not useful for fresh connections.
        for<'c> F: Fn(&'c mut DB::Connection, PoolConnectionMetadata) -> BoxFuture<'c, Result<(), Error>>
            + 'static
            + Send
            + Sync,
    {
        self.after_connect = Some(Box::new(callback));
        self
    }

    /// Perform an asynchronous action on a previously idle connection before giving it out.
    ///
    /// Alongside the connection, the closure gets [`PoolConnectionMetadata`] which contains
    /// potentially useful information such as the connection's age and the duration it was
    /// idle.
    ///
    /// If the operation returns `Ok(true)`, the connection is returned to the task that called
    /// [`Pool::acquire`].
    ///
    /// If the operation returns `Ok(false)` or an error, the error is logged (if applicable)
    /// and then the connection is closed and [`Pool::acquire`] tries again with another idle
    /// connection. If it runs out of idle connections, it opens a new connection instead.
    ///
    /// This is *not* invoked for new connections. Use [`after_connect`][Self::after_connect]
    /// for those.
    ///
    /// # Example: Custom `test_before_acquire` Logic
    /// If you only want to ping connections if they've been idle a certain amount of time,
    /// you can implement your own logic here:
    ///
    /// This example is written for Postgres but should be trivially adaptable to other databases.
    /// ```no_run
    /// # async fn f() -> Result<(), Box<dyn std::error::Error>> {
    /// use sqlx::{Connection, Executor};
    /// use sqlx::postgres::PgPoolOptions;
    ///
    /// let pool = PgPoolOptions::new()
    ///     .test_before_acquire(false)
    ///     .before_acquire(|conn, meta| Box::pin(async move {
    ///         // One minute
    ///         if meta.idle_for.as_secs() > 60 {
    ///             conn.ping().await?;
    ///         }
    ///
    ///         Ok(true)
    ///     }))
    ///     .connect("postgres:// …").await?;
    /// # Ok(())
    /// # }
    ///```
    ///
    /// For a discussion on why `Box::pin()` is required, see [the type-level docs][Self].
    pub fn before_acquire<F>(mut self, callback: F) -> Self
    where
        for<'c> F: Fn(&'c mut DB::Connection, PoolConnectionMetadata) -> BoxFuture<'c, Result<bool, Error>>
            + 'static
            + Send
            + Sync,
    {
        self.before_acquire = Some(Box::new(callback));
        self
    }

    /// Perform an asynchronous action on a connection before it is returned to the pool.
    ///
    /// Alongside the connection, the closure gets [`PoolConnectionMetadata`] which contains
    /// potentially useful information such as the connection's age.
    ///
    /// If the operation returns `Ok(true)`, the connection is returned to the pool's idle queue.
    /// If the operation returns `Ok(false)` or an error, the error is logged (if applicable)
    /// and the connection is closed, allowing a task waiting on [`Pool::acquire`] to
    /// open a new one in its place.
    ///
    /// # Example (Postgres): Close Memory-Hungry Connections
    /// Instead of relying on [`max_lifetime`] to close connections, we can monitor
    /// their memory usage directly and close any that have allocated too much.
    ///
    /// Note that this is purely an example showcasing a possible use for this callback
    /// and may be flawed as it has not been tested.
    ///
    /// This example queries [`pg_backend_memory_contexts`](https://www.postgresql.org/docs/current/view-pg-backend-memory-contexts.html)
    /// which is only allowed for superusers.
    ///
    /// ```no_run
    /// # async fn f() -> Result<(), Box<dyn std::error::Error>> {
    /// use sqlx::{Connection, Executor};
    /// use sqlx::postgres::PgPoolOptions;
    ///
    /// let pool = PgPoolOptions::new()
    ///     // Let connections live as long as they want.
    ///     .max_lifetime(None)
    ///     .after_release(|conn, meta| Box::pin(async move {
    ///         // Only check connections older than 6 hours.
    ///         if meta.age.as_secs() < 6 * 60 * 60 {
    ///             return Ok(true);
    ///         }
    ///
    ///         let total_memory_usage: i64 = sqlx::query_scalar(
    ///             "select sum(used_bytes) from pg_backend_memory_contexts"
    ///         )
    ///         .execute(conn)
    ///         .await?;
    ///
    ///         // Close the connection if the backend memory usage exceeds 256 MiB.
    ///         Ok(total_memory_usage <= (2 << 28))
    ///     }))
    ///     .connect("postgres:// …").await?;
    /// # Ok(())
    /// # }
    pub fn after_release<F>(mut self, callback: F) -> Self
    where
        for<'c> F: Fn(&'c mut DB::Connection, PoolConnectionMetadata) -> BoxFuture<'c, Result<bool, Error>>
            + 'static
            + Send
            + Sync,
    {
        self.after_release = Some(Box::new(callback));
        self
    }

    /// Creates a new pool from this configuration and immediately establishes one connection.
    pub async fn connect(self, uri: &str) -> Result<Pool<DB>, Error> {
        self.connect_with(uri.parse()?).await
    }

    /// Creates a new pool from this configuration and immediately establishes one connection.
    pub async fn connect_with(
        self,
        options: <DB::Connection as Connection>::Options,
    ) -> Result<Pool<DB>, Error> {
        let shared = PoolInner::new_arc(self, options);

        init_min_connections(&shared).await?;

        Ok(Pool(shared))
    }

    /// Creates a new pool from this configuration and will establish a connections as the pool
    /// starts to be used.
    pub fn connect_lazy(self, uri: &str) -> Result<Pool<DB>, Error> {
        Ok(self.connect_lazy_with(uri.parse()?))
    }

    /// Creates a new pool from this configuration and will establish a connections as the pool
    /// starts to be used.
    pub fn connect_lazy_with(self, options: <DB::Connection as Connection>::Options) -> Pool<DB> {
        let shared = PoolInner::new_arc(self, options);

        let _ = spawn({
            let shared = Arc::clone(&shared);
            async move {
                let _ = init_min_connections(&shared).await;
            }
        });

        Pool(shared)
    }
}

async fn init_min_connections<DB: Database>(pool: &Arc<PoolInner<DB>>) -> Result<(), Error> {
    for _ in 0..cmp::max(pool.options.min_connections, 1) {
        let deadline = Instant::now() + pool.options.acquire_timeout;
        let permit = pool.semaphore.acquire(1).await;

        // this guard will prevent us from exceeding `max_size`
        if let Ok(guard) = pool.try_increment_size(permit) {
            // [connect] will raise an error when past deadline
            let conn = pool.connection(deadline, guard).await?;
            pool.release(conn);
        }
    }

    Ok(())
}

impl<DB: Database> Debug for PoolOptions<DB> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("PoolOptions")
            .field("max_connections", &self.max_connections)
            .field("min_connections", &self.min_connections)
            .field("connect_timeout", &self.acquire_timeout)
            .field("max_lifetime", &self.max_lifetime)
            .field("idle_timeout", &self.idle_timeout)
            .field("test_before_acquire", &self.test_before_acquire)
            .finish()
    }
}
