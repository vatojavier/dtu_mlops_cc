#!/bin/bash

docker run --name predict --rm \
    -v $(pwd)/models/from_docker/trained_modelV1.pt:/models/trained_modelV1.pt \
    -v $(pwd)/data/raw/test.npz:/test.npz \
    predict:latest \
    models/from_docker/trained_modelV1.pt \
    data/raw/test.npz

