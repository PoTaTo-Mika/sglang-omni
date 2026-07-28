use std::sync::Mutex;

use crate::error::RouterError;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum State {
    Starting,
    Serving,
    Draining,
    Stopped,
    Failed,
}

/// Authoritative legal process-state transitions shared with local health.
pub(crate) struct Lifecycle {
    state: Mutex<State>,
}

impl Lifecycle {
    pub(crate) fn starting() -> Self {
        Self {
            state: Mutex::new(State::Starting),
        }
    }

    pub(crate) fn enter_serving(&self) -> Result<(), RouterError> {
        self.transition(State::Starting, State::Serving)
    }

    pub(crate) fn enter_draining(&self) -> Result<(), RouterError> {
        let mut state = self.state.lock().map_err(|_| RouterError::Lifecycle)?;
        match *state {
            State::Serving => {
                *state = State::Draining;
                Ok(())
            }
            State::Draining => Ok(()),
            State::Starting | State::Stopped | State::Failed => Err(RouterError::Lifecycle),
        }
    }

    pub(crate) fn enter_stopped(&self) -> Result<(), RouterError> {
        self.transition(State::Draining, State::Stopped)
    }

    pub(crate) fn enter_failed(&self) -> Result<(), RouterError> {
        let mut state = self.state.lock().map_err(|_| RouterError::Lifecycle)?;
        match *state {
            State::Starting | State::Serving | State::Draining => {
                *state = State::Failed;
                Ok(())
            }
            State::Stopped | State::Failed => Err(RouterError::Lifecycle),
        }
    }

    pub(crate) fn is_live(&self) -> bool {
        self.state
            .lock()
            .is_ok_and(|state| *state == State::Serving)
    }

    fn transition(&self, from: State, to: State) -> Result<(), RouterError> {
        let mut state = self.state.lock().map_err(|_| RouterError::Lifecycle)?;
        if *state != from {
            return Err(RouterError::Lifecycle);
        }
        *state = to;
        Ok(())
    }
}
