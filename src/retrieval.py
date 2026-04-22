import torch
import numpy as np
import faiss
import cn_clip.clip as clip
from PIL import Image
from tqdm import tqdm

class CrossModalRetriever:
    def __init__(self, K=3, device="cuda"):
        """
        Cross-modal retrieval system for recipe generation.
        This class utilizes Chinese-CLIP (ViT-L/14)
        to implement a non-parametric memory bank for retrieving culinary priors.

        :param K: Number of culinary priors to retrieve (Recommended K=3 in the paper).
        :param device: Computing device .
        """
        self.K = K
        self.device = device
        
        print("Loading Chinese-CLIP-ViT-L/14 model")
        self.model, self.preprocess = clip.load_from_name("ViT-L-14", device=self.device)
        self.model.eval()
        
        self.recipe_database = []
        self.index = None

    def build_index(self, dataset):
        """
        Constructs the non-parametric memory bank from the training dataset D_train.
        Encoding recipe text features into a FAISS index.

        :param dataset: Instance of the C2MRDataset class.
        """
        print(f" Building cross-modal retrieval index for {len(dataset)} items")
        
        all_recipe_texts = []
        for i in range(len(dataset)):
            # Extract data directly from JSON structure to ensure compatibility
            item = dataset.data[i]
            
            dish_name = item.get('\ufeff菜名') or item.get('菜名') or item.get('dish_name') or '未知菜品'
            
            # Extract and clean ingredients to serve as retrieval anchors
            raw_ings_str = item.get('xia_recipeIngredient', '')
            raw_ings = raw_ings_str.split('；') if isinstance(raw_ings_str, str) else []
            clean_ings = [dataset._clean_ingredient(ing) for ing in raw_ings if len(dataset._clean_ingredient(ing)) >= 1]
            ing_str = ",".join(clean_ings)
            
            # Extract cooking instructions
            instructions = dataset._parse_list_string(item.get('xia_recipeInstructions', []))
            
            recipe_text = f"菜名：{dish_name}。食材：{ing_str}"
            all_recipe_texts.append(recipe_text)
            
            # Map index to full recipe metadata in memory
            self.recipe_database.append({
                'name': dish_name,
                'ingredients': ing_str,
                'instructions': instructions 
            })

        # Batch encode text features using Chinese-CLIP
        print("Pre-computing recipe text feature vectors")
        all_features = []
        batch_size = 64
        
        with torch.no_grad():
            for i in tqdm(range(0, len(all_recipe_texts), batch_size)):
                batch_texts = all_recipe_texts[i : i + batch_size]
                # Tokenize
                text_tokens = clip.tokenize(batch_texts).to(self.device)
                # Encode
                features = self.model.encode_text(text_tokens)
                # Normalization
                features /= features.norm(dim=-1, keepdim=True)
                all_features.append(features.cpu().numpy())

        all_features = np.concatenate(all_features, axis=0).astype('float32')

        # Construct FAISS index for high-speed similarity search
        dim = all_features.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(all_features)
        
        print(f"Index construction complete. Feature dimensionality: {dim}")

    def retrieve(self, query_image_path):
        """
        Executes the cross-modal retrieval process.
        Maps input image 'v' to the K most semantically similar reference recipes M_prior.

        :param query_image_path: Path to the query image (v).
        :return: List of top-K relevant recipe dictionaries.
        """
        if self.index is None:
            raise RuntimeError("Index not built. Please call build_index() first.")

        # 1. Preprocess and encode the query image v
        try:
            image = self.preprocess(Image.open(query_image_path)).unsqueeze(0).to(self.device)
        except Exception as e:
            print(f"Failed to load query image: {query_image_path}")
            return []

        with torch.no_grad():
            # Extract visual features E_v
            image_features = self.model.encode_image(image)
            # Normalization
            image_features /= image_features.norm(dim=-1, keepdim=True)
            query_vec = image_features.cpu().numpy().astype('float32')

        # 2. Search FAISS index for the top-K cosine similarities
        similarities, indices = self.index.search(query_vec, self.K)

        # 3. Retrieve raw metadata from the internal database
        results = []
        for idx in indices[0]:
            if idx < len(self.recipe_database):
                results.append(self.recipe_database[idx])
        
        return results
