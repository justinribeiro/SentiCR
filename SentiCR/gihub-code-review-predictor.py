#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

# Justin says: This is sort of like the old test runner, but I've cut this down
# to be able to reuse saved model for some additional speed (since I need to
# parse a lot of data)
#
# Buyer beware. There be dragons.
__author__ = "Justin Ribeiro <justin@justinribeiro.com>"

from SentiCR import SentiCR
from pathlib import Path

import pandas as pd
import time

sentiment_analyzer = SentiCR(
    pretrained_model="gh-trainingset-pr-code-reviews-2023-09-20-17:16:33.joblib"
)


jdr_data = Path("/work/src/SentiCR/SentiCR/")
gh_review_comments = jdr_data / "data-review-comments-raw.pkl"


def predict_gh():
    begin = time.time()
    df = pd.read_pickle(gh_review_comments)
    df.insert(2, "score", "")

    for idx, row in df.iterrows():
        predicationBegin = time.time()
        id = df.loc[idx, "review_comment_id"]
        review = df.loc[idx, "body_stripped"]
        sentiment = sentiment_analyzer.get_sentiment_polarity(review)

        df.loc[idx, "score"] = sentiment[0]
        predictionEnd = time.time()
        timer = predictionEnd - predicationBegin
        print(
            f"{timer:.4f}s {df.loc[idx, 'score']}: {id}: {(review[:50] + '...') if len(review) > 75 else review}"
        )

    end = time.time()
    totalPredictionTime = end - begin
    print(
        f"Total Prediction time {totalPredictionTime:.2f} seconds for {df.shape[0]} PR code reviews"
    )
    df.to_csv("./senticr_data_run.csv", header=True)


predict_gh()
