# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# TODO (https://github.com/GoogleCloudPlatform/python-docs-samples/issues/8253): remove old region tags
# [START cloud_sql_mysql_sqlalchemy_connect_tcp]
# [START cloud_sql_mysql_sqlalchemy_sslcerts]
# [START cloud_sql_mysql_sqlalchemy_connect_tcp_sslcerts]
import os

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession, create_async_engine, async_scoped_session

import pymysql
from google.cloud.sql.connector import Connector, IPTypes
from asyncio import current_task


db_user = os.environ["DB_USER"]  # e.g. 'my-db-user'
db_pass = os.environ["DB_PASS"]  # e.g. 'my-db-password'
db_name = os.environ["DB_NAME"]  # e.g. 'my-database'
db_port = os.environ["DB_PORT"]  # e.g. 3306

# [END cloud_sql_mysql_sqlalchemy_connect_tcp]
connect_args = {}

# For deployments that connect directly to a Cloud SQL instance without
# using the Cloud SQL Proxy, configuring SSL certificates will ensure the
# connection is encrypted.
if os.environ.get("DB_ROOT_CERT"):
    db_root_cert = os.environ["DB_ROOT_CERT"]  # e.g. '/path/to/my/server-ca.pem'
    db_cert = os.environ["DB_CERT"]  # e.g. '/path/to/my/client-cert.pem'
    db_key = os.environ["DB_KEY"]  # e.g. '/path/to/my/client-key.pem'

    ssl_args = {"ssl_ca": db_root_cert, "ssl_cert": db_cert, "ssl_key": db_key}
    connect_args = ssl_args


if "INSTANCE_CONNECTION_NAME" in os.environ.keys():
    INSTANCE_CONNECTION_NAME = os.environ[
        "INSTANCE_CONNECTION_NAME"
    ]  # e.g. 'project:region:instance'
    ip_type = IPTypes.PRIVATE if os.environ.get("PRIVATE_IP") else IPTypes.PUBLIC
    connector = Connector(ip_type)

    def getconn() -> pymysql.connections.Connection:
        conn: pymysql.connections.Connection = connector.connect(
            INSTANCE_CONNECTION_NAME,
            "pymysql",
            user=db_user,
            password=db_pass,
            db=db_name+'-dev',
        )
        return conn


    engine = create_async_engine(
        "mysql+aiomysql://",
        creator=getconn,
        # [START_EXCLUDE]
        # Pool size is the maximum number of permanent connections to keep.
        pool_size=5,
        # Temporarily exceeds the set pool_size if no connections are available.
        max_overflow=2,
        # The total number of concurrent connections for your application will be
        # a total of pool_size and max_overflow.
        # 'pool_timeout' is the maximum number of seconds to wait when retrieving a
        # new connection from the pool. After the specified amount of time, an
        # exception will be thrown.
        pool_timeout=30,  # 30 seconds
        # 'pool_recycle' is the maximum number of seconds a connection can persist.
        # Connections that live longer than the specified amount of time will be
        # re-established
        pool_recycle=1800,  # 30 minutes
        # [END_EXCLUDE]
    )

else:
    # [START cloud_sql_mysql_sqlalchemy_connect_tcp]
    from sqlalchemy.engine.url import URL

    db_host = os.environ[
        "INSTANCE_HOST"
    ]  # e.g. '127.0.0.1' ('172.17.0.1' if deployed to GAE Flex)
    engine = create_async_engine(
        # Equivalent URL:
        # mysql+pymysql://<db_user>:<db_pass>@<db_host>:<db_port>/<db_name>
        URL.create(
            drivername="mysql+aiomysql",
            username=db_user,
            password=db_pass,
            host=db_host,
            port=int(db_port),
            database=db_name+'-dev',
        ),
        # [END cloud_sql_mysql_sqlalchemy_connect_tcp]
        connect_args=connect_args,
        # [START cloud_sql_mysql_sqlalchemy_connect_tcp]
        # [START_EXCLUDE]
        # [START cloud_sql_mysql_sqlalchemy_limit]
        # Pool size is the maximum number of permanent connections to keep.
        pool_size=10,
        # Temporarily exceeds the set pool_size if no connections are available.
        max_overflow=2,
        # The total number of concurrent connections for your application will be
        # a total of pool_size and max_overflow.
        # [END cloud_sql_mysql_sqlalchemy_limit]
        # [START cloud_sql_mysql_sqlalchemy_backoff]
        # SQLAlchemy automatically uses delays between failed connection attempts,
        # but provides no arguments for configuration.
        # [END cloud_sql_mysql_sqlalchemy_backoff]
        # [START cloud_sql_mysql_sqlalchemy_timeout]
        # 'pool_timeout' is the maximum number of seconds to wait when retrieving a
        # new connection from the pool. After the specified amount of time, an
        # exception will be thrown.
        pool_timeout=30,  # 30 seconds
        # [END cloud_sql_mysql_sqlalchemy_timeout]
        # [START cloud_sql_mysql_sqlalchemy_lifetime]
        # 'pool_recycle' is the maximum number of seconds a connection can persist.
        # Connections that live longer than the specified amount of time will be
        # re-established
        pool_recycle=1800,  # 30 minutes
        # [END cloud_sql_mysql_sqlalchemy_lifetime]
        # [END_EXCLUDE]
    )

AsyncSessionMaker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
AsyncSessionScoped = async_scoped_session(AsyncSessionMaker, scopefunc=current_task)
Base = declarative_base()


# Dependency
async def get_session() -> AsyncSession:
    async with AsyncSessionScoped() as session:
        yield session

# [END cloud_sql_mysql_sqlalchemy_connect_tcp_sslcerts]
# [END cloud_sql_mysql_sqlalchemy_sslcerts]
# [END cloud_sql_mysql_sqlalchemy_connect_tcp]


if __name__ == "__main__":
    db = get_session()