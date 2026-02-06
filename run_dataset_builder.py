from app.pipelines.data_pipeline import DataPipeline
from app.core.dataset_builder import TimeSeriesDatasetBuilder

if __name__ == "__main__":

    df = DataPipeline().run()

    builder = TimeSeriesDatasetBuilder()

    X_train, X_test, y_train, y_test = builder.prepare(df)

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)
