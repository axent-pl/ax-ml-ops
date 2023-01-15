#!/usr/bin/env bash
./stop.sh
for service in $(docker-compose ps -q)
do
  volumes=$(docker inspect -f '{{.Mounts}}' ${service})
  docker rm -v $service
  echo $volumes | tr "{" "\n" | while read volume;
  do
    if [[ $volume == volume* ]]; then
      volume=$(echo $volume | cut -d" " -f2)
      docker volume rm $volume
    fi
  done
done
