import json
import os
import requests
import hashlib
from PIL import Image
import re
import ast

class C2MRDataset:
    """
    Dataset loader for C2MR (Chinese Recipe Multi-modal dataset).
    Handles JSON parsing, image downloading/caching, and text cleaning for recipe data.
    
    :param json_path: Path to the dataset JSON file.
    :param image_cache_dir: Directory to store downloaded recipe images locally.
    """
    def __init__(self, json_path="data/C2MR.json", image_cache_dir="./image_cache"):
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found: {json_path}")
            
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
        self.image_cache_dir = image_cache_dir
        if not os.path.exists(self.image_cache_dir):
            os.makedirs(self.image_cache_dir)
            
        self.all_ingredients = set()
        print(f"Initializing dataset ({len(self.data)} items) and building ingredient library")
        
        # Build a global set of unique cleaned ingredients from the raw data
        for item in self.data:
            # Compatible with specific JSON key "xia_recipeIngredient"
            raw_ings_str = item.get('xia_recipeIngredient', '')
            # Splits format like "Pork:300g; Potato:150g" using Chinese semicolon
            raw_ings = raw_ings_str.split('；') if isinstance(raw_ings_str, str) else []
            clean_ings = [self._clean_ingredient(ing) for ing in raw_ings]
            # Filter out empty strings or invalid entries
            clean_ings = [ing for ing in clean_ings if len(ing) >= 1]
            self.all_ingredients.update(clean_ings)
            
    def _clean_ingredient(self, text):
        """
        Removes quantities, units, and non-Chinese characters from ingredient strings.
        
        :param text: Raw ingredient string (e.g., "Pork: 500g").
        :return: Cleaned ingredient name (e.g., "Pork").
        """
        if not isinstance(text, str): return ""
        # Remove anything after separators like colons or spaces
        text = re.split(r'[:：\s]', text)[0]
        # Remove alphanumeric characters and dots
        text = re.sub(r'[0-9a-zA-Z\.]', '', text)
         # Filter out common Chinese measurement units and quantifiers
        text = re.sub(r'(适量|少许|半个|一个|一勺|两勺|克|两|斤|只|个|块|片|把|滴|份|大|小|长|短|粗|细)', '', text)
        # Keep only Chinese characters
        text = re.sub(r'[^\u4e00-\u9fa5]', '', text)
        return text.strip()

    def _parse_list_string(self, val):
        """
        Safely converts string representations of lists into Python list objects.
        
        :param val: Input which could be a list or a stringified list.
        :return: Python list.
        """
        if isinstance(val, list):
            return val
        try:
            cleaned_str = str(val).replace('“', '"').replace('”', '"')
            return ast.literal_eval(cleaned_str)
        except:
            return []

    def _get_or_download_image(self, url):
        """
        Downloads image from URL or retrieves it from local cache using MD5 hashing.
        
        :param url: Image URL.
        :return: Tuple of (PIL.Image object, local_file_path).
        """
        if not url or not str(url).startswith('http'):
            return None, None
            
        url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
        file_path = os.path.join(self.image_cache_dir, f"{url_hash}.jpg")
        
        if not os.path.exists(file_path):
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                with open(file_path, 'wb') as f:
                    f.write(response.content)
            except Exception as e:
                return None, None
                
        try:
            img = Image.open(file_path).convert("RGB")
            return img, file_path 
        except Exception as e:
            return None, None

    def get_recipe_data(self, index):
        """
        Main interface to extract processed recipe components by index.
        
        :param index: Index of the item in the dataset.
        :return: Tuple of (Image, Local_Path, Ingredients_List, Instructions_List, Metadata).
        """
        item = self.data[index]
        
        # Fetch and cache the dish image
        img_url = item.get('图片地址')
        v_image, v_local_path = self._get_or_download_image(img_url)
        
        # Process and clean ingredient list
        raw_ings_str = item.get('xia_recipeIngredient', '')
        raw_ings = raw_ings_str.split('；') if isinstance(raw_ings_str, str) else []
        I_ingredients = [self._clean_ingredient(ing) for ing in raw_ings if len(self._clean_ingredient(ing)) >= 1]
        
        # Parse cooking instructions into a list
        S_instructions = self._parse_list_string(item.get('xia_recipeInstructions', []))
        
         # Retrieve dish name, handling potential Byte Order Mark (BOM) or different key variations
        dish_name = item.get('\ufeff菜名') or item.get('菜名') or item.get('dish_name') or '未知菜品'
        
        metadata = {
            'dish_name': dish_name,
            'cuisine': item.get('主分类', item.get('分类名称', '中餐'))
        }
        
        return v_image, v_local_path, I_ingredients, S_instructions, metadata

    def __len__(self):
        return len(self.data)