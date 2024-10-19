import polars as pl
from pathlib import Path
DIFF_QUANTILE = 0.75 # TODO: optimize
MIN_SECONDS_PER_PERSON = 2
SHORT_SEGMENTS_SECONDS_THRESH = 0.3

def add_diffs(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        [pl.col(x).diff().abs().alias(f"{x}_diff") for x in df.columns if "x_" in x or "y_" in x]
    )


def replace_zeros_with_nans(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        [pl.col(x).replace(0, None) for x in df.columns if "x_" in x or "y_" in x]
    )

def filter_by_percentile(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(pl.col("max_diff") > pl.col("max_diff").quantile(DIFF_QUANTILE))

def filter_by_datapoints_per_person(df: pl.DataFrame, fps: int) -> pl.DataFrame:
    """remove persons with appearence shorter than MIN_SECONDS_PER_PERSON"""
    return df.filter(
        pl.col('person').is_in(
            df['person'].value_counts().filter(pl.col('count')> fps*MIN_SECONDS_PER_PERSON)['person']
        )
    )
def filter_values(df: pl.DataFrame, fps:int) -> pl.DataFrame:
    return (
        df.filter(
        # no movement
        (pl.col('max_diff')>0)
        # extreme movement
        & (pl.col('max_diff')<(pl.col('max_diff').mean()+2*pl.col('max_diff').std()))
        ).with_columns(frame_diff=pl.col("frame").diff()).filter(
        # short segments
        pl.col('frame_diff')<fps*SHORT_SEGMENTS_SECONDS_THRESH
        )
    )
    

def save_chart(df: pl.DataFrame, output_path: Path, name: str) -> pl.DataFrame:
    df.plot.line(x="frame", y="y_Right Knee", color="person").save(output_path/ f"{name}.png")
    return df

def save_df(df, output_path: Path, name: str, save=False,) -> None:
    "save a parquet file with all of the extracted skeleton keypoints"
    if save:
        df.write_parquet(output_path / f"{name}.parquet")
    return df

def process_keypoints(df: pl.DataFrame, output_path: Path, fps:int, save_debug=False) -> pl.DataFrame:
    df = (
        df
        .pivot(on="keypoint", values=["x", "y"], index=["person", "frame"])
        .filter(pl.col("frame")!=0)
        .pipe(replace_zeros_with_nans)
        .sort(["person", "frame"])
        .fill_null(strategy="forward")
        .with_columns(pl.col("person").cast(pl.Utf8))
        .pipe(add_diffs)
        .pipe(save_df, output_path, "all_keypoints", save_debug)
        .pipe(save_chart, output_path, "before_filtering")
    )
    # keep frames that have a lot of movement
    df = (
        df
        .with_columns(max_diff=pl.sum_horizontal([pl.col(x) for x in df.columns if "_diff" in x]))
        .pipe(filter_by_datapoints_per_person, fps)
        .pipe(filter_values, fps)
        .pipe(save_chart, output_path, "after_filtering")
        .select(['person', 'frame', 'max_diff'])
        .pipe(filter_by_percentile)
        .unique('frame')
        .sort("frame")
        .with_columns(frame_diff=pl.col("frame").diff())
        .pipe(save_df, output_path, "processed_keypoints_data", save_debug)
    )
    return df

