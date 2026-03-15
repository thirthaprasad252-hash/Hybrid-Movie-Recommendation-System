"""
Hybrid Movie Recommendation System
====================================
Production-grade implementation suitable for ML engineering portfolios.
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="CineMatch — Hybrid Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ─────────────────────────────────────────
st.markdown("""
<style>
.metric-card {background:#1e293b;border-radius:10px;padding:20px;text-align:center}
.rec-card {background:#1e293b;border-left:4px solid #f59e0b;border-radius:8px;padding:12px;margin:6px 0}
.badge {background:#f59e0b22;color:#f59e0b;padding:2px 8px;border-radius:20px;font-size:12px}
.section-header {color:#f59e0b;font-weight:700;letter-spacing:2px}
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════
# RECOMMENDER ENGINE
# ═════════════════════════════════════════════════════
class RecommenderEngine:

    def __init__(self, movies, ratings, n_components=50):

        self.movies = movies.copy()
        self.ratings = ratings.copy()
        self.n_components = n_components

        self._preprocess()
        self._build_content_model()
        self._build_cf_model()
        self._build_popularity_index()
        self.rmse = self._evaluate()

    # ── Preprocess ─────────────────────────
    def _preprocess(self):

        self.movies['genres'] = self.movies['genres'].replace("(no genres listed)", "Unknown")
        self.movies['genres_clean'] = self.movies['genres'].str.replace("|"," ",regex=False)
        self.movies['year'] = self.movies['title'].str.extract(r'\((\d{4})\)').astype(float)

        self.title_to_idx = pd.Series(self.movies.index,index=self.movies['title']).drop_duplicates()

    # ── Content Model ──────────────────────
    def _build_content_model(self):

        tfidf = TfidfVectorizer(stop_words="english")
        tfidf_matrix = tfidf.fit_transform(self.movies['genres_clean'])

        self.cosine_sim = cosine_similarity(tfidf_matrix,tfidf_matrix)

    def content_recommend(self,title,top_n=10):

        idx = self.title_to_idx[title]

        scores = list(enumerate(self.cosine_sim[idx]))
        scores = sorted(scores,key=lambda x:x[1],reverse=True)[1:top_n+1]

        indices=[i[0] for i in scores]
        sims=[round(i[1],3) for i in scores]

        result=self.movies.iloc[indices][['title','genres','year']].copy()
        result['similarity']=sims
        return result.reset_index(drop=True)

    # ── Collaborative Model ─────────────────
    def _build_cf_model(self):

        self.user_movie=self.ratings.pivot(
            index="userId",
            columns="movieId",
            values="rating"
        ).fillna(0)

        svd=TruncatedSVD(n_components=self.n_components,random_state=42)

        self.latent=svd.fit_transform(self.user_movie)
        self.items=svd.components_

        self.reconstructed=self.latent@self.items

    def cf_recommend(self,user_id,top_n=10):

        if user_id not in self.user_movie.index:
            return pd.DataFrame()

        user_idx=list(self.user_movie.index).index(user_id)

        pred=self.reconstructed[user_idx]

        rated=self.user_movie.loc[user_id].values>0
        pred=np.where(rated,-np.inf,pred)

        top_cols=np.argsort(pred)[::-1][:top_n]

        movie_ids=self.user_movie.columns[top_cols]

        result=self.movies[self.movies['movieId'].isin(movie_ids)][['title','genres','year','movieId']]

        result['cf_score']=pred[top_cols]
        return result.drop(columns='movieId').reset_index(drop=True)

    # ── Popularity Model ────────────────────
    def _build_popularity_index(self):

        stats=self.ratings.groupby('movieId').agg(
            count=('rating','count'),
            mean=('rating','mean')
        )

        C=stats['count'].mean()
        m=stats['mean'].mean()

        stats['score']=(stats['count']/(stats['count']+C))*stats['mean'] + (C/(stats['count']+C))*m

        self.popularity=stats.sort_values('score',ascending=False).reset_index()

        self.popularity=self.popularity.merge(
            self.movies[['movieId','title','genres','year']],on='movieId'
        )

    def popular_recommend(self,top_n=10):

        return self.popularity.head(top_n)[['title','genres','year','score','count','mean']]

    # ── Hybrid ─────────────────────────────
    def hybrid_recommend(self,user_id,title,top_n=10,alpha=0.5):

        cf=self.cf_recommend(user_id,top_n*3)
        cb=self.content_recommend(title,top_n*3)

        if cf.empty:
            return cb.head(top_n)

        cf['norm']=(cf['cf_score']-cf['cf_score'].min())/(cf['cf_score'].max()-cf['cf_score'].min()+1e-9)
        cb['norm']=cb['similarity']

        merged=pd.merge(
            cf[['title','genres','year','norm']],
            cb[['title','norm']],
            on='title',
            suffixes=('_cf','_cb'),
            how='outer'
        ).fillna(0)

        merged['hybrid_score']=alpha*merged['norm_cf']+(1-alpha)*merged['norm_cb']

        return merged.sort_values('hybrid_score',ascending=False).head(top_n)

    # ── Evaluation ─────────────────────────
    def _evaluate(self):

        train,test=train_test_split(self.ratings,test_size=0.2,random_state=42)

        matrix=train.pivot(index='userId',columns='movieId',values='rating').fillna(0)

        svd=TruncatedSVD(n_components=self.n_components)

        latent=svd.fit_transform(matrix)
        recon=latent@svd.components_

        preds=[]
        actuals=[]

        for _,row in test.iterrows():

            if row['userId'] in matrix.index and row['movieId'] in matrix.columns:

                u=list(matrix.index).index(row['userId'])
                m=list(matrix.columns).index(row['movieId'])

                preds.append(recon[u,m])
                actuals.append(row['rating'])

        return round(np.sqrt(mean_squared_error(actuals,preds)),4)

    # ── Stats ──────────────────────────────
    @property
    def dataset_stats(self):

        return {
            "Movies":len(self.movies),
            "Ratings":len(self.ratings),
            "Users":self.ratings['userId'].nunique(),
            "RMSE":self.rmse
        }

    @property
    def genre_distribution(self):

        return self.movies['genres'].str.split('|').explode().value_counts().head(15)

    @property
    def rating_distribution(self):

        return self.ratings['rating'].value_counts().sort_index()


# ═════════════════════════════════════════════════════
# DATA
# ═════════════════════════════════════════════════════

@st.cache_data
def load_data():
    movies=pd.read_csv("movies.csv")
    ratings=pd.read_csv("ratings.csv")
    return movies,ratings


@st.cache_resource
def build_engine(n_components):
    movies,ratings=load_data()
    return RecommenderEngine(movies,ratings,n_components)


# ═════════════════════════════════════════════════════
# UI
# ═════════════════════════════════════════════════════

st.title("🎬 CineMatch")

with st.sidebar:

    n_components=st.slider("SVD Components",10,100,50)

    engine=build_engine(n_components)

    mode=st.radio("Mode",["Hybrid","Content-Based","Collaborative","Popularity"])

    user_id=st.number_input("User ID",1,int(engine.ratings['userId'].max()),1)

    title=st.selectbox("Seed Movie",engine.movies['title'])

    top_n=st.slider("Recommendations",3,20,8)

    if mode=="Hybrid":
        alpha=st.slider("Alpha",0.0,1.0,0.5)

    btn=st.button("Recommend")


tab1,tab2=st.tabs(["Recommendations","EDA"])


# ═════════════════════════════════════════════════════
# RECOMMENDATIONS
# ═════════════════════════════════════════════════════

with tab1:

    if btn:

        if mode=="Hybrid":
            recs=engine.hybrid_recommend(user_id,title,top_n,alpha)

        elif mode=="Content-Based":
            recs=engine.content_recommend(title,top_n)

        elif mode=="Collaborative":
            recs=engine.cf_recommend(user_id,top_n)

        else:
            recs=engine.popular_recommend(top_n)

        st.dataframe(recs)


# ═════════════════════════════════════════════════════
# EDA
# ═════════════════════════════════════════════════════

with tab2:

    st.subheader("Genre Distribution")
    st.bar_chart(engine.genre_distribution)

    col1,col2=st.columns(2)

    with col1:

        st.subheader("Rating Distribution")
        st.bar_chart(engine.rating_distribution)

    with col2:

        st.subheader("Ratings per User Histogram")

        user_counts=engine.ratings.groupby('userId').size()

        bins=pd.cut(
            user_counts,
            bins=[0,25,50,100,200,500,3000]
        ).value_counts().sort_index()

        # ✅ FIXED HERE
        bins.index=bins.index.astype(str)

        st.bar_chart(bins)

    st.subheader("Top Rated Movies")

    st.dataframe(
        engine.popular_recommend(10)
    )