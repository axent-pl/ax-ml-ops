FROM minio/mc:RELEASE.2022-07-06T14-54-36Z

COPY setup.sh /opt/dap/
RUN chmod +x /opt/dap/setup.sh

ENTRYPOINT [ "/bin/sh" ]
CMD ["/opt/dap/setup.sh"]