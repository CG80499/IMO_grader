import tinker
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, force=True)
service_client = tinker.ServiceClient()
print("Available models:")
for item in service_client.get_server_capabilities().supported_models:
    print("- " + item.model_name)
