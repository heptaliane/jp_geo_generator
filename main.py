#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from typing import Annotated, Literal

import click
import numpy as np
import numpy.typing as npt
import requests
import skimage.measure
from stl import mesh

EQUATOR_LENGTH = 40000
GEO_SIZE = 256
GEO_ERR_VALUE = -1
GEO_SOURCE_URL = "https://cyberjapandata.gsi.go.jp/xyz/dem/{z}/{x}/{y}.txt"


def get_z_scale(z: int) -> float:
    xy_length = EQUATOR_LENGTH / z
    return GEO_SIZE / xy_length


class GeoDataProvider:
    def __init__(self, cache_path: str = ".cache.npz"):
        self._cache_path = cache_path
        self._cache = {}
        if os.path.exists(cache_path):
            self._cache = dict(np.load(cache_path))

    def _download_geo_data(
        self,
        x: int,
        y: int,
        z: int,
    ) -> Annotated[npt.NDArray[np.float32], Literal[GEO_SIZE, GEO_SIZE]]:
        url = GEO_SOURCE_URL.format(x=x, y=y, z=z)
        res = requests.get(url)

        if res.status_code == 404:
            return np.full((GEO_SIZE, GEO_SIZE), GEO_ERR_VALUE, dtype=np.float32)
        elif res.status_code != 200:
            raise ValueError(res.content)

        arr = [
            [GEO_ERR_VALUE if h == "e" else float(h) for h in line.split(",")]
            for line in res.content.decode("utf-8").split("\n")
            if len(line) > 0
        ]
        return np.asarray(arr, dtype=np.float32)

    def get(
        self,
        x: int,
        y: int,
        z: int,
    ) -> Annotated[npt.NDArray[np.float32], Literal[GEO_SIZE, GEO_SIZE]]:
        key = f"{x}-{y}-{z}"

        if key not in self._cache:
            data = self._download_geo_data(x, y, z)
            self._cache[key] = data
            np.savez_compressed(self._cache_path, **self._cache)

        return self._cache[key]


def concat_geo_data(
    data: list[list[Annotated[npt.NDArray[np.float32], Literal[GEO_SIZE, GEO_SIZE]]]]
) -> Annotated[npt.NDArray[np.float32], Literal["N", "N"]]:
    return np.block(data)


def sample_data(
    data: Annotated[npt.NDArray[np.float32], Literal["N", "N"]],
    sample_rate: int,
) -> Annotated[npt.NDArray[np.float32], Literal["N", "N"]]:
    return skimage.measure.block_reduce(data, (sample_rate, sample_rate), np.max)


def create_index_matrix(
    data: Annotated[npt.NDArray[np.float32], Literal["N", "N"]],
) -> Annotated[npt.NDArray[np.uint32], Literal["N", "N"]]:
    xlen, ylen = data.shape

    return np.arange(xlen * ylen).reshape(xlen, ylen)


def create_vertices(
    data: Annotated[npt.NDArray[np.float32], Literal["N", "N"]],
) -> Annotated[npt.NDArray[np.uint32], Literal["N", "N", 3]]:
    _, ylen = data.shape
    indices = create_index_matrix(data)

    vertices = np.transpose(
        np.stack((indices // ylen, indices % ylen, data)),
        axes=(1, 2, 0),
    ).reshape(-1, 3)

    bottom_vertices = np.copy(vertices)
    bottom_vertices[:, 2] = 0

    return np.concat((vertices, bottom_vertices))


def create_triangles(
    indices: Annotated[npt.NDArray[np.uint32], Literal["N"]],
    offsets: Annotated[npt.NDArray[np.uint32], Literal[3]],
) -> Annotated[npt.NDArray[np.uint32], Literal["N", 3]]:
    vertices = list()
    for offset in offsets:
        vertices.append(indices + offset)

    return np.vstack(vertices).T


def create_polygon(
    data: Annotated[npt.NDArray[np.float32], Literal["N", "N"]],
) -> Annotated[npt.NDArray[np.float32], Literal["N", 3, 3]]:
    vertices = create_vertices(data)
    padded = np.pad(data, (0, 1), constant_values=(0, GEO_ERR_VALUE))

    val_mask_0 = padded[:-1, :-1] != GEO_ERR_VALUE
    val_mask_1 = padded[:-1, 1:] != GEO_ERR_VALUE
    val_mask_2 = padded[1:, :-1] != GEO_ERR_VALUE
    val_mask_3 = padded[1:, 1:] != GEO_ERR_VALUE

    tri_mask_013 = val_mask_0 & val_mask_1 & val_mask_3
    tri_mask_023 = val_mask_0 & val_mask_2 & val_mask_3
    tri_mask_012 = val_mask_0 & val_mask_1 & val_mask_2 & np.logical_not(val_mask_3)
    tri_mask_123 = np.logical_not(val_mask_0) & val_mask_1 & val_mask_2 & val_mask_3

    edge_tri_mask_013 = tri_mask_013 & np.logical_not(val_mask_2)
    edge_tri_mask_023 = tri_mask_023 & np.logical_not(val_mask_1)

    edge_mask_03 = edge_tri_mask_013 | edge_tri_mask_023
    edge_mask_12 = tri_mask_012 | tri_mask_123

    val_mask_01 = val_mask_0 & val_mask_1
    val_mask_02 = val_mask_0 & val_mask_2
    val_mask_13 = val_mask_1 & val_mask_3
    val_mask_23 = val_mask_2 & val_mask_3

    edge_mask_01 = val_mask_01 & np.logical_not(val_mask_23)
    edge_mask_02 = val_mask_02 & np.logical_not(val_mask_13)
    edge_mask_13 = val_mask_13 & np.logical_not(val_mask_02)
    edge_mask_23 = val_mask_23 & np.logical_not(val_mask_01)

    indices = create_index_matrix(data)
    offsets = indices[:2, :2].flatten()
    offsets = np.concat((offsets, offsets + indices[-1, -1] + 1))

    triangles = np.concatenate(
        [
            # Top Surfaces
            create_triangles(indices[tri_mask_013], offsets[[0, 1, 3]]),
            create_triangles(indices[tri_mask_023], offsets[[0, 2, 3]]),
            create_triangles(indices[tri_mask_012], offsets[[0, 1, 2]]),
            create_triangles(indices[tri_mask_123], offsets[[1, 2, 3]]),
            # Bottom Surfaces
            create_triangles(indices[tri_mask_013], offsets[[4, 5, 7]]),
            create_triangles(indices[tri_mask_023], offsets[[4, 6, 7]]),
            create_triangles(indices[tri_mask_012], offsets[[4, 5, 6]]),
            create_triangles(indices[tri_mask_123], offsets[[5, 6, 7]]),
            # Edge Surfaces
            create_triangles(indices[edge_mask_03], offsets[[0, 3, 4]]),
            create_triangles(indices[edge_mask_03], offsets[[3, 4, 7]]),
            create_triangles(indices[edge_mask_12], offsets[[1, 2, 5]]),
            create_triangles(indices[edge_mask_12], offsets[[2, 5, 6]]),
            create_triangles(indices[edge_mask_01], offsets[[0, 1, 4]]),
            create_triangles(indices[edge_mask_01], offsets[[1, 4, 5]]),
            create_triangles(indices[edge_mask_02], offsets[[0, 2, 4]]),
            create_triangles(indices[edge_mask_02], offsets[[2, 4, 6]]),
            create_triangles(indices[edge_mask_13], offsets[[1, 3, 5]]),
            create_triangles(indices[edge_mask_13], offsets[[3, 5, 7]]),
            create_triangles(indices[edge_mask_23], offsets[[2, 3, 6]]),
            create_triangles(indices[edge_mask_23], offsets[[3, 6, 7]]),
            create_triangles(indices[0, :][val_mask_01[0, :]], offsets[[0, 1, 4]]),
            create_triangles(indices[0, :][val_mask_01[0, :]], offsets[[1, 4, 5]]),
            create_triangles(indices[:, 0][val_mask_02[:, 0]], offsets[[0, 2, 4]]),
            create_triangles(indices[:, 0][val_mask_02[:, 0]], offsets[[2, 4, 6]]),
        ]
    )

    return vertices[triangles]


@click.command()
@click.option("--x", type=int, required=True)
@click.option("--y", type=int, required=True)
@click.option("--z", type=int, default=12)
@click.option("--offset", type=int, default=10)
@click.option("--output", type=str, default="geo.stl")
@click.option("--blocks", nargs=2, type=int, default=(1, 1))
@click.option("--size", type=float, default=50.0)
@click.option("--sample_rate", type=int, default=1)
def main(
    x: int,
    y: int,
    z: int,
    offset: int,
    output: str,
    blocks: tuple[int, int],
    size: float,
    sample_rate: int,
):
    geo_provider = GeoDataProvider()

    xblock, yblock = blocks
    data = [
        [geo_provider.get(xi, yj, z) for xi in range(x, x + xblock, 1)]
        for yj in range(y, y + yblock, 1)
    ]
    concat_data = concat_geo_data(data)

    z_scale = get_z_scale(z)
    valid_mask = concat_data != GEO_ERR_VALUE
    concat_data[valid_mask] = concat_data[valid_mask] * z_scale + offset

    compressed_data = sample_data(concat_data, sample_rate)
    mesh_data = create_polygon(compressed_data)
    scale = size / max(compressed_data.shape)
    mesh_data *= scale

    geo_mesh = mesh.Mesh(np.zeros(mesh_data.shape[0], dtype=mesh.Mesh.dtype))
    geo_mesh.remove_duplicate_polygons = True
    geo_mesh.vectors = mesh_data

    geo_mesh.save(output)


if __name__ == "__main__":
    main()
