# Federated Learning System - Step-by-Step Guide

This README provides a visual walkthrough of the federated learning training and aggregation loop between a cloud server and Jetson client devices.

## Step 1: Start the Cloud Script
On the cloud (host PC), run the following command:
```bash
python cloud_ori.py
```
![Step 1](figs/ins2.1.png)

## Step 2: Start the Client Training Script
On each Jetson client, execute:
```bash
python train_ori.py
```
![Step 2](ins3.2.png)

## Step 3: Cloud Waits for Upload
The cloud waits for client uploads to `receive/` folder:
![Step 3](ins3.3.png)

## Step 4: Jetson Begins Training
Training starts based on the assigned dataset (e.g. js11):
![Step 4](ins3.4.png)

## Step 5: Training Progress
Jetson shows training progress with accuracy and loss values:
![Step 5](ins3.5.png)

## Step 6: Upload to Cloud
After training, Jetson uploads `.h5` model and `.txt` log:
![Step 6](ins3.6.png)

## Step 7: Cloud Issues Next Instruction
Cloud sends new instruction to begin round 2 (e.g., js12, js22):
![Step 7](ins3.7.png)

## Step 8: Training New Round
Jetson now trains using the next dataset (e.g., js12):
![Step 8](ins3.8.png)

## Step 9: Repeating the Cycle
The cloud continues the loop for all rounds:
![Step 9](ins3.9.png)
