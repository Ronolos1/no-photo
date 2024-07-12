from pymongo import MongoClient
import os
import requests
import pandas as pd

# MongoDB connection setup
connection_string = 'mongodb+srv://officialreca0:eswxP6699ruM1jWT@cluster0.yhis61q.mongodb.net/?retryWrites=true&w=majority'
client = MongoClient(connection_string)

# Product groups
product_groups = {
    "group1": ["668930c15e724b282ae05021", "668930d25e724b282ae0502c", "66892d115e724b282ae04fe9"],
    "group2": ["668e6801247a3aad812934f6", "6689aea7bb3871963fbb6d9d", "6689aec7bb3871963fbb6da3", "6689aef0bb3871963fbb6da9", "6689af14bb3871963fbb6db4"],
    "group3": ["6689ac4cbb3871963fbb6d93", "6689ac11bb3871963fbb6d7d", "6689ac31bb3871963fbb6d88", "668931f15e724b282ae05037", "668932125e724b282ae05063"]
}

# Function to load data from MongoDB into a DataFrame
def load_data_from_mongo(db_name, collection_name):
    db = client[db_name]
    collection = db[collection_name]
    data = list(collection.find())
    df = pd.DataFrame(data)
    return df

# Function to get products data
def get_products_data(db_name, collection_name='products'):
    return load_data_from_mongo(db_name, collection_name)

# Function to download images for products
def download_product_images(db_name, recon_path='./product_images/recon'):
    products_df = get_products_data(db_name)

    if not os.path.exists(recon_path):
        os.makedirs(recon_path)

    print("Starting image download process...")

    for group_name, product_ids in product_groups.items():
        group_path = os.path.join(recon_path, group_name)

        if not os.path.exists(group_path):
            os.makedirs(group_path)

        for product_id in product_ids:
            product = products_df[products_df['_id'].astype(str) == product_id].iloc[0]
            images = product['images']  # List of image URLs

            print(f"Downloading images for product '{product_id}' in group '{group_name}'")

            for i, image_url in enumerate(images):
                image_name = f"{product_id}_{i+1}.jpg"  # Naming convention: productID_1.jpg, productID_2.jpg, ...
                image_path = os.path.join(group_path, image_name)

                # Check if image file already exists
                if os.path.exists(image_path):
                    print(f"Skipping download for {image_name}. Image already exists.")
                    continue

                # Download image using requests
                try:
                    response = requests.get(image_url, stream=True)
                    if response.status_code == 200:
                        with open(image_path, 'wb') as file:
                            for chunk in response.iter_content(chunk_size=1024):
                                file.write(chunk)
                        print(f"Downloaded: {image_name}")
                    else:
                        print(f"Failed to download image {image_url}. Status code: {response.status_code}")
                except Exception as e:
                    print(f"Error downloading image {image_url}: {e}")

    print("Image download process completed.")

if __name__ == "__main__":
    # Example usage to download images
    download_product_images('test', './product_images/recon')
