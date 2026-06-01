class BaseAgent:
    def act(self, obs):
        raise NotImplementedError

    def update(self, batch):
        raise NotImplementedError
