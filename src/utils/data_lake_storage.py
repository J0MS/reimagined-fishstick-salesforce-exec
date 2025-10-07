from azure.storage.filedatalake import DataLakeServiceClient
from io import StringIO
import os

class DataLakeStorage:
    """
    A class to interact with Azure Data Lake Storage, allowing for file reading
    and writing operations.

    Attributes:
        file_system_client: A client to interact with a specific file system (container) in Azure Data Lake.
    """

    def __init__(self):
        """
        Initializes the DeltaStorage instance by creating a client for Azure Data Lake Storage.

        This constructor fetches configuration details from environment variables
        and establishes a connection to the specified container.
        """
        ACCOUNT_NAME = os.getenv("DATA_LAKE_STORAGE_ACCOUNT_NAME", "")
        CONTAINER_NAME = os.getenv("DATA_LAKE_STORAGE_CONTAINER_NAME", "")
        ACCOUNT_KEY = os.getenv("DATA_LAKE_STORAGE_ACCOUNT_KEY", "")

        # Create a DataLakeServiceClient using the account name and key.
        service_client = DataLakeServiceClient(account_url=f"https://{ACCOUNT_NAME}.dfs.core.windows.net", credential=ACCOUNT_KEY)

        # Get a file system client for the specified container.
        self.file_system_client = service_client.get_file_system_client(CONTAINER_NAME)

    def read_file(self, file_path: str) -> StringIO:
        """
        Reads the content of a file from Azure Data Lake Storage.

        Args:
            file_path (str): The path of the file to read from the Data Lake.

        Returns:
            StringIO: An in-memory string buffer containing the file content.
        """
        file_client = self.file_system_client.get_file_client(file_path)
        downloaded_file = file_client.download_file()
        file_content = downloaded_file.readall().decode('utf-8')
        return StringIO(file_content)

    def write_file(self, data, file_path: str) -> None:
        """
        Writes data to a specified file in Azure Data Lake Storage.

        Args:
            data (str): The content to write to the file.
            file_path (str): The path of the file where data will be written.

        This method overwrites the file if it already exists.
        """
        file_client = self.file_system_client.get_file_client(file_path)
        file_client.upload_data(data, overwrite=True)


data_lake_storage = DataLakeStorage()
