"""
read data from json and saved to df
"""

import hydra
from omegaconf import DictConfig
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import json
import pandas as pd
from sklearn.preprocessing import MinMaxScaler 

@hydra.main(config_path="../config", config_name="main", version_base=None)
def process_data(config: DictConfig):
    """Function to process the data"""
    
    print(f"Process data using {config.data.raw}")
    print(f"Columns used: {config.process.use_columns}")
    playlist_data = json.load(open("../data/raw/spotify_million_playlist_dataset/data/mpd.slice.0-999.json"))
    playlists = playlist_data["playlists"]
    df = pd.DataFrame(playlists)

    # Rename the columns to match the desired column names
    df.rename(columns={"name": "playlist_name"}, inplace=True)

    # Explode the track data to create separate rows for each track
    df = df.explode("tracks")
    print(df.head(3))
    print(df['tracks'].head(3))
    # Create separate columns for "track_name," "track_uri," and "artist_name"
    expand_columns = list(playlist_data["playlists"][0]['tracks'][0].keys())
    print(expand_columns)
    df[expand_columns] = df["tracks"].apply(pd.Series)

    # Drop the original "track_data" column
    df.drop(columns=["tracks"], inplace=True)
    #why are we using pySpark
    df.reset_index(inplace=True, drop=True)
    df.to_pickle("../data/processed/playlist.pkl")


def extract_audio_feature(df, k=100):
    '''
    df: dataframe contains a column of "track_uri" 
    process this by 100 per time due to the API limit 

    return: extended dataframe with audio feature columns
    '''
    features = []
    #TODO: fine tune this 
    batch = df.shape[0]//k + 1
    for i in range(batch):
        melt = df.iloc[i*100: i*100 + 100]['track_uri']
        melt = melt.to_list()
        features = features + sp.audio_features(melt)
    #last batch - updated i 
    if df.shape[0]%k != 0: 
        #TODO: remove this 
        melt=df.iloc[i*100:]['track_uri']
        melt = melt.to_list()
        features = features + sp.audio_features(melt)
    feature_df = pd.DataFrame(features)
    rows = feature_df.shape[0]
    songs = df.iloc[:rows+1]
    songs = pd.concat([songs, feature_df], axis=1)
    return songs

def extract_sentiment_score(df):
    """
    df: a dataframe contains column 'track_name' 
    output: a extended dataframe that contains 2 new column 'subjectivity' and 'polarity'
    """
    df['polarity'] = df['track_name'].astype(str).apply(lambda name: TextBlob(name).sentiment.polarity)
    df['subjectivity'] = df['track_name'].astype(str).apply(lambda name: TextBlob(name).sentiment.subjectivity)
    return df


def scale_feature(df):
    """
    Scale numeric features for songs in db
    output: df scaled
    """
    scaler = MinMaxScaler()
    #numeric cols:
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    # Returning the scaled dataframe
    return df


if __name__ == "__main__":
    cid = '8846ee3eb88148599920d8096724f7bd'
    secret = 'a60d08bbbaf540a5bb1da93285da5d5a'
    client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
    sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)
    process_data()
