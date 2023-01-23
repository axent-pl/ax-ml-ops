#!/usr/bin/bash
docker-compose down
docker volume ls | grep $(pwd | rev | cut -d"/" -f1 | rev) | sed 's/  */ /' | cut -d " " -f2 | xargs docker volume rm