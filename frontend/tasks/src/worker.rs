//! A model on its own thread. Its runner cannot move.

use crate::Error;
use std::sync::{Mutex, mpsc};

type Work<T> = Box<dyn FnOnce(&mut T) + Send>;

/// Runs calls on `T` one at a time.
pub struct Worker<T> {
    work: Mutex<mpsc::Sender<Work<T>>>,
}

impl<T: 'static> Worker<T> {
    /// Calls `open` on a new thread, and waits for it.
    pub fn spawn(open: impl FnOnce() -> Result<T, Error> + Send + 'static) -> Result<Self, Error> {
        let (work, queue) = mpsc::channel::<Work<T>>();
        let (ready, opened) = mpsc::channel();
        let scope = tessera_engine::check::current_scope();
        std::thread::spawn(move || {
            let result = match scope {
                Some(scope) => scope.run(open),
                None => open(),
            };
            match result {
                Ok(mut model) => {
                    let _ = ready.send(Ok(()));
                    queue.iter().for_each(|w| w(&mut model));
                }
                Err(e) => drop(ready.send(Err(e))),
            }
        });
        opened.recv().map_err(|_| "the model thread stopped")??;
        Ok(Self {
            work: Mutex::new(work),
        })
    }

    /// Runs `f` on the thread and waits for it.
    pub fn run<R: Send + 'static>(
        &self,
        f: impl FnOnce(&mut T) -> R + Send + 'static,
    ) -> Result<R, Error> {
        let (answer, answered) = mpsc::channel();
        let scope = tessera_engine::check::current_scope();
        let work = self.work.lock().map_err(|_| "the model stopped")?;
        work.send(Box::new(move |m| {
            let out = match scope {
                Some(scope) => scope.run(|| f(m)),
                None => f(m),
            };
            drop(answer.send(out));
        }))
        .map_err(|_| "the model stopped")?;
        drop(work);
        Ok(answered.recv().map_err(|_| "the model stopped")?)
    }
}
