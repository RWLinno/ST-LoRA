#!/bin/bash
cd data

echo "Downloading Metr-LA datasets..."
gdown https://drive.google.com/u/0/uc?id=1vvNDeYD6k7Srm6SaVtqEOFohagA1Jxa-
tar -xvf metrla.tar
rm metrla.tar

echo "Downloading PEMSBAY datasets..."
gdown https://drive.google.com/u/0/uc?id=1jcogPdX1As9Zxt_u0hLGW0ym97HWMF-t
tar -xvf pemsbay.tar
rm pemsbay.tar

echo "Downloading pems03 datasets..."
gdown https://drive.google.com/u/0/uc?id=1F1GU9Xk-2vr03BE3LudGKYvbXoCBoJMJ
tar -xvf pems03.tar
rm pems03.tar

echo "Downloading pems04 datasets..."
gdown https://drive.google.com/u/0/uc?id=1Q-ldmCDW3a9Rb8zBKGd8XcSmpSghyrIx
tar -xvf pems04.tar
rm pems04.tar

echo "Downloading pems07 datasets..."
gdown https://drive.google.com/u/0/uc?id=1VmWibz1Ba7Wq-60s_5vjfV1yHhlpuzFy
tar -xvf pems07.tar
rm pems07.tar

echo "Downloading pems08 datasets..."
gdown https://drive.google.com/u/0/uc?id=1T_GMeyrKEYz269q8Q6SpgrarB1EEjIG4
tar -xvf pems08.tar
rm pems08.tar