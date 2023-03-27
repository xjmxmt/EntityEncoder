import numpy as np
import pandas as pd
from category_encoders import CatBoostEncoder
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from entityencoder_utils.general_utils import hardcoded_log


class Normalizer:
    def __init__(self, mins: pd.Series, maxs: pd.Series):
        """Normalize values to between -1 and 1
        """
        self.mins = mins
        self.maxs = maxs

    def transform(self, values: pd.DataFrame) -> pd.DataFrame:
        values = (values - self.mins) / (self.maxs - self.mins)
        values = values * 2 - 1
        return values

    def reverse_transform(self, values: np.ndarray) -> np.ndarray:
        inverted = (values + 1) / 2
        inverted = inverted * (self.maxs.values - self.mins.values) + self.mins.values
        return inverted


class GroupByPCAEmbedding:
    def __init__(
            self,
            data: pd.DataFrame,
            categorical_cols: list
    ):
        """
        Embeddings generated by applying aggregate function on other columns

        :param data: Dataframe
        :param categorical_cols: list of categorical columns
        """

        self.categorical_cols = categorical_cols

        self.embedding_sizes = []
        self.pca_list = []
        self.normalizer_list = []
        self.nn_classifier_list = []
        self.cat_embs_list = []
        for col in categorical_cols:
            print(f'log: processing column: {col}')
            values = data[col]

            n_uniq = values.nunique()
            embedding_size = hardcoded_log(n_uniq)
            self.embedding_sizes.append((n_uniq, embedding_size))

            # drop our target categorical column
            cat_embs = data.loc[values.index].drop(columns=col)

            # transform the other categorical columns to something more meaningful
            # (TODO can try ordinal or one-hot or anything else from category_encoders here, what performs best?)
            cat_embs = CatBoostEncoder().fit_transform(cat_embs, values.astype("category").cat.codes)

            # concatenate target column back to encoded dataframe
            cat_embs = pd.concat((values, cat_embs), axis=1)

            # get the actual fingerprints for each category in our target column
            # (TODO which aggregations work best here? should we also add in different quantiles? or some other functions?)
            cat_embs = cat_embs.groupby(col).agg(["min", "median", "max", "mean", "std"]).fillna(0)
            indices = cat_embs.index

            # standardize the embeddings for PCA (drop any columns with no variation)
            cat_embs = cat_embs - cat_embs.mean(0)
            cat_embs = cat_embs.drop(columns=cat_embs.columns[cat_embs.std(0) == 0])  # drop constant columns
            cat_embs = cat_embs / cat_embs.std(0)

            # encode the large fingerprint vectors to the fixed embedding size
            # TODO you can grab the eigenvectors from the pca for federated learning with self.pca.components_
            pca = PCA(n_components=embedding_size).fit(cat_embs.values)
            cat_embs = pd.DataFrame(pca.transform(cat_embs.values), index=indices)

            # normalize embedding values to between -1 and 1 (TODO maybe apply the Quantile Transform here?)
            mins, maxs = cat_embs.min(), cat_embs.max()
            normalizer = Normalizer(mins, maxs)
            self.normalizer_list.append(normalizer)

            # train a nearest neighbor classifier to invert our embedding back to categories
            nn_classifier = KNeighborsClassifier(n_neighbors=1).fit(cat_embs, np.arange(len(cat_embs)))

            self.pca_list.append(pca)
            self.nn_classifier_list.append(nn_classifier)
            self.cat_embs_list.append(cat_embs)
        self.total_emb_size = sum([i for _, i in self.embedding_sizes])

    def transform(self, values: pd.DataFrame) -> np.ndarray:
        embeddings_array = []
        for idx, col_name in enumerate(values.columns):
            cat_embs = self.cat_embs_list[idx]
            column = values[col_name]

            # select the embedding vector for each category
            embeddings = cat_embs.loc[column]

            normalizer = self.normalizer_list[idx]
            transformed = normalizer.transform(embeddings)
            embeddings_array.append(transformed)

        embeddings_array = np.hstack(embeddings_array)
        return embeddings_array

    def inverse(self, values: np.ndarray) -> np.ndarray:
        assert values.shape[1] == self.total_emb_size
        i = 0
        rec_cat = []
        for idx, (n_uniq, embedding_size) in enumerate(self.embedding_sizes):
            cat_embs = values[:, i:i+embedding_size]
            i += embedding_size

            # revert transformation back to regular embedding space
            normalizer = self.normalizer_list[idx]
            inverted = normalizer.reverse_transform(cat_embs)

            # predict the most likely category based on the embedding
            nn_classifier = self.nn_classifier_list[idx]
            indices = nn_classifier.predict(inverted)

            cat_embs = self.cat_embs_list[idx]
            rec_cat.append(cat_embs.index[indices])

        rec_cat = np.hstack(rec_cat)
        return rec_cat


if __name__ == "__main__":

    cover = pd.read_csv("../test_cases/data/cover/processed_cover.csv")
    categorical = [col for col, dtype in cover.convert_dtypes().dtypes.items() if str(dtype) in ["string", "Int64"]]
    cover[categorical] = cover[categorical].astype("category")
    C = GroupByPCAEmbedding(cover, categorical)

    enc = C.transform(cover[categorical])
    print(enc.shape, "\n")
    dec = C.inverse(enc)
    print(dec, "\n")
