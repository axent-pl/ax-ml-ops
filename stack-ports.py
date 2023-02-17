#!/usr/bin/python

import yaml

if __name__ == '__main__':
    with open('docker-compose.yaml') as dc:
        y = yaml.safe_load(dc)
        for service in sorted(y['services']):
            if 'ports' in y['services'][service]:
                print(service, y['services'][service]['ports'])