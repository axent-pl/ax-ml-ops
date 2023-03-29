class ModelRegistry:
    models = {}

    @classmethod
    def register(self,cls):
        self.models[cls.__name__] = cls

    @classmethod
    def is_registered(self, model_name):
        return model_name in self.models
    
    @classmethod
    def get(self, model_name):
        return self.models[model_name]