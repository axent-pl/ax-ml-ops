set -e
echo "[INFO ]: Configuring connection alias"
mc alias set dap http://${S3_HOST}:${S3_PORT} ${S3_AUTH_ROOT_KEY_ID} ${S3_AUTH_ROOT_SECRET_KEY}
until mc ls dap &> /dev/null; do echo "[INFO ]: Waiting for S3..."; done;
echo "[INFO ]: Creating bucket"
mc mb -p dap/${S3_BUCKET}
echo "[INFO ]: Adding user"
mc admin user add dap ${S3_AUTH_KEY_ID} ${S3_AUTH_SECRET_KEY}
echo "[INFO ]: Set readwrite policy for user"
mc admin policy set dap readwrite user=${S3_AUTH_KEY_ID}
echo "[INFO ]: Done"