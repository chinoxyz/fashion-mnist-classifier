def preprocess_min_max(features, min_value, max_value):
    return (features - min_value) / (max_value - min_value)
