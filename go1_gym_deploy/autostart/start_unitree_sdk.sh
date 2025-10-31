#!/bin/bash
sudo docker stop foxy_controller || true
sudo docker rm foxy_controller || true
lsof -ti:8090 | xargs kill -9 || true
sudo kill $(ps aux |grep lcm_position | awk '{print $2}')
sudo fuser -k 7667/udp || true
sudo fuser -k 8090/udp || true
sleep 1
cd ~/go1_gym/go1_gym_deploy/unitree_legged_sdk_bin/
yes "" | sudo ./lcm_position &