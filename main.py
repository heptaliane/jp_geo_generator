#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from typing import Annotated, Literal

import click
import numpy as np
import numpy.typing as npt
import requests
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
            self._cache = np.load(cache_path)

    def _download_geo_data(
        self,
        x: int,
        y: int,
        z: int,
    ) -> Annotated[npt.NDArray[np.float32], Literal[GEO_SIZE, GEO_SIZE]]:
        url = GEO_SOURCE_URL.format(x=x, y=y, z=z)
        res = requests.get(url)

        if res.status_code != 200:
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
    return np.concatenate([np.concatenate(row) for row in data])


def create_index_matrix(
    data: Annotated[npt.NDArray[np.float32], Literal["N", "N"]],
) -> Annotated[npt.NDArray[np.uint32], Literal["N", "N"]]:
    xlen, ylen = data.shape

    return np.arange(xlen * ylen).reshape(xlen, ylen)


def create_vertices(
    data: Annotated[npt.NDArray[np.float32], Literal["N", "N"]],
) -> Annotated[npt.NDArray[np.uint32], Literal["N", "N", 3]]:
    xlen, _ = data.shape
    indices = create_index_matrix(data)
    vertices = np.stack((indices % xlen, indices // xlen, data))

    return np.transpose(vertices, axes=(1, 2, 0)).reshape(-1, 3)


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
    padded = np.pad(data, (0, 1), constant_values=(0, GEO_ERR_VALUE))

    val_mask_00 = padded[:-1, :-1] != GEO_ERR_VALUE
    val_mask_01 = padded[:-1, 1:] != GEO_ERR_VALUE
    val_mask_10 = padded[1:, :-1] != GEO_ERR_VALUE
    val_mask_11 = padded[1:, 1:] != GEO_ERR_VALUE

    tri_mask_013 = val_mask_00 & val_mask_01 & val_mask_11
    tri_mask_023 = val_mask_00 & val_mask_10 & val_mask_11
    tri_mask_012 = val_mask_00 & val_mask_01 & val_mask_10 & np.logical_not(val_mask_11)
    tri_mask_123 = np.logical_not(val_mask_00) & val_mask_01 & val_mask_10 & val_mask_11

    indices = create_index_matrix(data)

    offsets = indices[:2, :2].flatten()
    triangles = np.concatenate(
        [
            create_triangles(indices[tri_mask_013], offsets[[0, 1, 3]]),
            create_triangles(indices[tri_mask_023], offsets[[0, 2, 3]]),
            create_triangles(indices[tri_mask_012], offsets[[0, 1, 2]]),
            create_triangles(indices[tri_mask_123], offsets[[1, 2, 3]]),
        ]
    )

    vertices = create_vertices(data)
    return vertices[triangles]


@click.command()
@click.option("--x", type=int, required=True)
@click.option("--y", type=int, required=True)
@click.option("--z", type=int, default=12)
@click.option("--offset", type=int, default=10)
@click.option("--output", type=str, default="geo.stl")
@click.option("--size", type=float, default=1.0)
def main(
    x: int,
    y: int,
    z: int,
    offset: int,
    output: str,
    size: float,
):
    geo_provider = GeoDataProvider()
    data = geo_provider.get(x, y, z)

    z_scale = get_z_scale(z)
    data[data != GEO_ERR_VALUE] *= z_scale

    mesh_data = create_polygon(data)

    geo_mesh = mesh.Mesh(np.zeros(mesh_data.shape[0], dtype=mesh.Mesh.dtype))
    geo_mesh.remove_duplicate_polygons = True
    geo_mesh.vectors = mesh_data

    geo_mesh.save(output)


if __name__ == "__main__":
    main()
