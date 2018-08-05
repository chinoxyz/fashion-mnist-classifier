class SampleGenerator():
    def __init__(self, features, target):
        self.features = features
        self.target = target

    def get_batches(self, batch_size):
        limit = int((len(self.features) + batch_size - 1) / batch_size)

        for i in range(limit):
            l = i * batch_size
            u = min((i+1) * batch_size, len(self.features))
            yield self.features[l:u], self.target[l:u]

    def get_num_samples(self):
        return self.target.size


class SampleSet():
    def __init__(self, features, target, validation_ratio):
        training_set_size = int(target.size * (1 - validation_ratio))
        features_training = features[:training_set_size]
        target_training = target[:training_set_size]

        self.training_sample_generator = SampleGenerator(features_training, target_training)
        features_validation = features[training_set_size:]
        target_validation = target[training_set_size:]

        self.validation_sample_generator = SampleGenerator(features_validation, target_validation)

    def get_training_batches(self, batch_size):
        return self.training_sample_generator.get_batches(batch_size)

    def get_validation_batches(self, batch_size):
        return self.validation_sample_generator.get_batches(batch_size)
