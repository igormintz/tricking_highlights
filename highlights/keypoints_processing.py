import polars as pl
from pathlib import Path
DIFF_QUANTILE = 0.85
FRAME_DIFF = 40

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

def save_intermediate_csv(df: pl.DataFrame, output_path: Path) -> pl.DataFrame:
    df.write_parquet(output_path/ "processed_keypoints_data.parquet")
    return df

def save_chart(df: pl.DataFrame, output_path: Path) -> pl.DataFrame:
    df.plot.line(x="frame", y="y_Right Knee", color="person").save(output_path/ "chart.png")
    return df

def process_keypoints(df: pl.DataFrame, output_path: Path) -> pl.DataFrame:
    df = (
        df
        .pivot(on="keypoint", values=["x", "y"], index=["person", "frame"])
        .filter(pl.col("frame")!=0)
        .pipe(replace_zeros_with_nans)
        .sort(["person", "frame"])
        .fill_null(strategy="forward")
        .with_columns(pl.col("person").cast(pl.Utf8))
        .pipe(add_diffs)
        .pipe(save_chart, output_path)
    )
    df = (
        df
        .with_columns(max_diff=pl.sum_horizontal([pl.col(x) for x in df.columns if "_diff" in x]))
        .select(['person', 'frame', 'max_diff'])
        .pipe(filter_by_percentile)
        .unique('frame')
        .sort("frame")
        .with_columns(frame_diff=pl.col("frame").diff())
        .pipe(save_intermediate_csv, output_path)
        .filter(pl.col("frame_diff")<=FRAME_DIFF)
    )
    return df

