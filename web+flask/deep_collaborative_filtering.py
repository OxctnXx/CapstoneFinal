import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from gensim.models import Word2Vec
import jieba
import matplotlib.pyplot as plt
from collections import defaultdict

class RecommenderModel(nn.Module):
    """PyTorch model for deep collaborative filtering"""
    def __init__(self, n_users, n_items, embedding_dim, dropout_rate):
        super(RecommenderModel, self).__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        self.description_dense = nn.Linear(300, embedding_dim)
        
        self.fc1 = nn.Linear(embedding_dim * 3, 64)
        self.fc2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        
    def forward(self, user_input, item_input, description_input):
        user_embedded = self.dropout(self.user_embedding(user_input).squeeze(1))
        item_embedded = self.dropout(self.item_embedding(item_input).squeeze(1))
        
        description_dense = self.dropout(self.relu(self.description_dense(description_input)))
        
        concatenated = torch.cat([user_embedded, item_embedded, description_dense], dim=1)
        
        dense1 = self.dropout(self.relu(self.fc1(concatenated)))
        dense2 = self.dropout(self.relu(self.fc2(dense1)))
        
        output = self.output(dense2)
        
        return output

class DeepCollaborativeFiltering:
    def __init__(self, menu_file, orders_file, embedding_dim=32, dropout_rate=0.2, l2_reg=1e-5):
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.menu_data = self._load_json(menu_file)
        self.orders_data = self._load_json(orders_file)
        
        self.all_items = self._extract_all_items()
        self.user_item_matrix, self.users, self.items = self._create_user_item_matrix()
        
        self.n_users = len(self.users)
        self.n_items = len(self.items)
        
        self.description_embeddings = self._create_description_embeddings()
        self.item_flavor_profiles = self._create_item_flavor_profiles()
        
        self.model = RecommenderModel(self.n_users, self.n_items, embedding_dim, dropout_rate).to(self.device)
        self.trained = False
    
    def _load_json(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _extract_all_items(self):
        all_items = []
        
        for category in self.menu_data["drinks"]:
            for item in self.menu_data["drinks"][category]:
                all_items.append(item)
        
        for item in self.menu_data["cakes"]:
            all_items.append(item)
            
        return all_items
    
    def _create_user_item_matrix(self):
        users = set()
        items = set()
        
        for order in self.orders_data["orders"]:
            users.add(order["customer_id"])
            for item in order["items"]:
                items.add(item["id"])
        
        users = sorted(list(users))
        items = sorted(list(items))
        
        self.user_to_index = {user: i for i, user in enumerate(users)}
        self.item_to_index = {item: i for i, item in enumerate(items)}
        self.index_to_user = {i: user for i, user in enumerate(users)}
        self.index_to_item = {i: item for i, item in enumerate(items)}
        
        matrix = np.zeros((len(users), len(items)))
        
        for order in self.orders_data["orders"]:
            user_idx = self.user_to_index[order["customer_id"]]
            for item in order["items"]:
                if item["id"] in self.item_to_index:
                    item_idx = self.item_to_index[item["id"]]
                    matrix[user_idx, item_idx] += item.get("quantity", 1)
        
        self.user_item_pairs = []
        self.ratings = []
        
        for u_idx in range(len(users)):
            for i_idx in range(len(items)):
                if matrix[u_idx, i_idx] > 0:
                    self.user_item_pairs.append((u_idx, i_idx))
                    self.ratings.append(matrix[u_idx, i_idx])
        
        return matrix, users, items
    
    def _create_description_embeddings(self):
        print("Creating description embeddings...")
        
        descriptions = []
        item_ids = []
        
        for item in self.all_items:
            if "description" in item:
                segmented_desc = ' '.join(jieba.cut(item["description"]))
                descriptions.append(segmented_desc)
                item_ids.append(item["id"])
        
        vectorizer = TfidfVectorizer(max_features=300)
        tfidf_matrix = vectorizer.fit_transform(descriptions)
        
        description_embeddings = {}
        for i, item_id in enumerate(item_ids):
            description_embeddings[item_id] = tfidf_matrix[i].toarray()[0]
            
        default_vector = np.zeros(300)
        
        for item_id in self.items:
            if item_id not in description_embeddings:
                description_embeddings[item_id] = default_vector
                
        print(f"Created description embeddings for {len(description_embeddings)} items")
        return description_embeddings
    
    def _extract_flavor_keywords(self, text):
        flavor_keywords = [
            "sweet", "bitter", "sour", "salty", "spicy", "mellow", "fragrant", "rich", "refreshing", "crispy",
            "dense", "caramel", "chocolate", "cream", "milk", "fruity", "tea", "coffee",
            "silky", "soft", "cheese", "thick", "mango", "strawberry", "coconut", "berry", "peach",
            "fresh", "grapefruit", "floral", "fruit", "nutty", "matcha", "tropical", "flavor"
        ]
        
        found_keywords = []
        for keyword in flavor_keywords:
            if keyword in text:
                found_keywords.append(keyword)
                
        return found_keywords
    
    def _create_item_flavor_profiles(self):
        flavor_profiles = {}
        
        for item in self.all_items:
            item_id = item["id"]
            description = item.get("description", "")
            name = item.get("name", "")
            
            text = name + " " + description
            flavor_keywords = self._extract_flavor_keywords(text)
            flavor_profiles[item_id] = flavor_keywords
            
        return flavor_profiles
    
    def train(self, epochs=20, batch_size=64, validation_split=0.2, verbose=1):
        if not self.user_item_pairs:
            print("No interaction data to train on")
            return
        
        users = np.array([pair[0] for pair in self.user_item_pairs])
        items = np.array([pair[1] for pair in self.user_item_pairs])
        ratings = np.array(self.ratings)
        
        descriptions = np.array([
            self.description_embeddings[self.index_to_item[item_idx]] 
            for item_idx in items
        ])
        
        if descriptions.shape[1] != 300:
            print(f"Warning: Description embeddings have shape {descriptions.shape}")
            new_descriptions = np.zeros((descriptions.shape[0], 300))
            for i, desc in enumerate(descriptions):
                new_descriptions[i, :min(desc.shape[0], 300)] = desc[:min(desc.shape[0], 300)]
            descriptions = new_descriptions
            print(f"Adjusted descriptions to shape {descriptions.shape}")
        
        users_tensor = torch.LongTensor(users)
        items_tensor = torch.LongTensor(items)
        ratings_tensor = torch.FloatTensor(ratings)
        descriptions_tensor = torch.FloatTensor(descriptions)
        
        dataset = TensorDataset(users_tensor, items_tensor, descriptions_tensor, ratings_tensor)
        
        total_samples = len(dataset)
        train_size = int((1 - validation_split) * total_samples)
        val_size = total_samples - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=self.l2_reg)
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            for batch_users, batch_items, batch_descriptions, batch_ratings in train_loader:
                batch_users = batch_users.to(self.device)
                batch_items = batch_items.to(self.device)
                batch_descriptions = batch_descriptions.to(self.device)
                batch_ratings = batch_ratings.to(self.device).view(-1, 1)
                
                outputs = self.model(batch_users, batch_items, batch_descriptions)
                loss = criterion(outputs, batch_ratings)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * batch_users.size(0)
            
            train_loss /= train_size
            
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_users, batch_items, batch_descriptions, batch_ratings in val_loader:
                    batch_users = batch_users.to(self.device)
                    batch_items = batch_items.to(self.device)
                    batch_descriptions = batch_descriptions.to(self.device)
                    batch_ratings = batch_ratings.to(self.device).view(-1, 1)
                    
                    outputs = self.model(batch_users, batch_items, batch_descriptions)
                    loss = criterion(outputs, batch_ratings)
                    
                    val_loss += loss.item() * batch_users.size(0)
                
                val_loss /= val_size
            
            if verbose:
                print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}')
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pth')
        
        self.model.load_state_dict(torch.load('best_model.pth'))
        self.trained = True
    
    def predict_rating(self, user_id, item_id):
        if not self.trained:
            print("Model not trained yet")
            return 0
        
        if user_id not in self.user_to_index or item_id not in self.item_to_index:
            return 0
        
        user_idx = self.user_to_index[user_id]
        item_idx = self.item_to_index[item_id]
        
        description_vector = self.description_embeddings[item_id]
        
        if len(description_vector) != 300:
            new_vector = np.zeros(300)
            new_vector[:min(len(description_vector), 300)] = description_vector[:min(len(description_vector), 300)]
            description_vector = new_vector
        
        user_tensor = torch.LongTensor([user_idx]).to(self.device)
        item_tensor = torch.LongTensor([item_idx]).to(self.device)
        description_tensor = torch.FloatTensor([description_vector]).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(user_tensor, item_tensor, description_tensor)
        
        return float(prediction.cpu().numpy()[0][0])
    
    def _get_item_info(self, item_id):
        for category in self.menu_data["drinks"]:
            for item in self.menu_data["drinks"][category]:
                if item["id"] == item_id:
                    return item
        
        for item in self.menu_data["cakes"]:
            if item["id"] == item_id:
                return item
                
        return None
    
    def get_user_recommendations(self, user_id, top_n=5, exclude_purchased=True):
        if not self.trained:
            print("Model not trained yet")
            return []
        
        if user_id not in self.user_to_index:
            print(f"User {user_id} not found in the dataset")
            return []
            
        user_idx = self.user_to_index[user_id]
        
        purchased_items = set()
        if exclude_purchased:
            purchased_items = set(np.where(self.user_item_matrix[user_idx] > 0)[0])
        
        predicted_ratings = []
        
        for item_idx in range(self.n_items):
            if item_idx in purchased_items:
                continue
                
            item_id = self.index_to_item[item_idx]
            rating = self.predict_rating(user_id, item_id)
            
            if rating > 0:
                predicted_ratings.append((item_id, rating))
        
        predicted_ratings.sort(key=lambda x: x[1], reverse=True)
        
        top_recommendations = []
        
        for item_id, rating in predicted_ratings[:top_n]:
            item_info = self._get_item_info(item_id)
            
            if item_info:
                item_info = item_info.copy()
                item_info["prediction_score"] = rating
                explanation = self._generate_explanation(user_id, item_id)
                item_info["explanation"] = explanation
                top_recommendations.append(item_info)
        
        return top_recommendations
    
    def _generate_explanation(self, user_id, item_id):
        user_idx = self.user_to_index[user_id]
        item_idx = self.item_to_index[item_id]
        item_info = self._get_item_info(item_id)
        
        if not item_info:
            return "Cannot generate recommendation explanation."
            
        purchased_items = np.where(self.user_item_matrix[user_idx] > 0)[0]
        
        if len(purchased_items) == 0:
            return "Based on your taste preferences, we think you might like this choice."
            
        similarities = []
        
        for p_item_idx in purchased_items:
            p_item_id = self.index_to_item[p_item_idx]
            p_item_info = self._get_item_info(p_item_id)
            
            if p_item_info:
                if item_id in self.description_embeddings and p_item_id in self.description_embeddings:
                    desc_sim = cosine_similarity(
                        [self.description_embeddings[item_id]], 
                        [self.description_embeddings[p_item_id]]
                    )[0][0]
                else:
                    desc_sim = 0
                    
                item_flavors = set(self.item_flavor_profiles.get(item_id, []))
                p_item_flavors = set(self.item_flavor_profiles.get(p_item_id, []))
                
                if item_flavors and p_item_flavors:
                    flavor_sim = len(item_flavors.intersection(p_item_flavors)) / len(item_flavors.union(p_item_flavors))
                else:
                    flavor_sim = 0
                
                cat_sim = 1.0 if item_info.get("category") == p_item_info.get("category") else 0.0
                combined_sim = 0.4 * desc_sim + 0.4 * flavor_sim + 0.2 * cat_sim
                similarities.append((p_item_id, combined_sim, p_item_info))
                
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        if not similarities:
            return "Based on your taste preferences, we think you might like this choice."
            
        most_similar_id, similarity, most_similar_info = similarities[0]
        item_flavors = set(self.item_flavor_profiles.get(item_id, []))
        similar_flavors = set(self.item_flavor_profiles.get(most_similar_id, []))
        common_flavors = item_flavors.intersection(similar_flavors)
        
        explanation = f"Recommendation reason: "
        
        if similarity > 0.7:
            explanation += f"You liked {most_similar_info['name']} which is very similar to this"
        elif similarity > 0.4:
            explanation += f"Based on your purchase of {most_similar_info['name']}"
        else:
            explanation += "Based on your historical preferences"
            
        if common_flavors:
            explanation += f", both have {', '.join(common_flavors)} characteristics"
            
        if item_info.get("category") == most_similar_info.get("category"):
            explanation += f", both are {item_info.get('category')} category"
            
        if "description" in item_info:
            highlight = item_info["description"][:15] + "..."
            explanation += f", {highlight}"
            
        return explanation

    def analyze_flavor_preferences(self, user_id):
        if user_id not in self.user_to_index:
            print(f"User {user_id} not found in the dataset")
            return {}
        
        user_idx = self.user_to_index[user_id]
        purchased_item_indices = np.where(self.user_item_matrix[user_idx] > 0)[0]
        
        if len(purchased_item_indices) == 0:
            return {}
            
        flavor_counts = defaultdict(float)
        total_items = 0
        
        for item_idx in purchased_item_indices:
            rating = self.user_item_matrix[user_idx, item_idx]
            item_id = self.index_to_item[item_idx]
            flavor_keywords = self.item_flavor_profiles.get(item_id, [])
            
            for keyword in flavor_keywords:
                flavor_counts[keyword] += rating
                
            total_items += rating
            
        flavor_preferences = {k: (v / total_items) * 100 for k, v in flavor_counts.items()}
        
        return flavor_preferences
    
    def explain_with_flavor_preferences(self, user_id, item_id):
        user_preferences = self.analyze_flavor_preferences(user_id)
        
        if not user_preferences:
            return "Cannot generate explanation based on flavor preferences."
            
        item_flavors = self.item_flavor_profiles.get(item_id, [])
        
        if not item_flavors:
            return "Cannot generate explanation based on flavor preferences."
            
        matching_flavors = []
        for flavor in item_flavors:
            if flavor in user_preferences and user_preferences[flavor] > 0:
                matching_flavors.append((flavor, user_preferences[flavor]))
                
        matching_flavors.sort(key=lambda x: x[1], reverse=True)
        
        if not matching_flavors:
            return "This is a new experience based on your existing taste preferences."
            
        explanation = "Based on your taste preferences,"
        
        if len(matching_flavors) == 1:
            explanation += f" you prefer {matching_flavors[0][0]} flavored items,"
        elif len(matching_flavors) == 2:
            explanation += f" you prefer {matching_flavors[0][0]} and {matching_flavors[1][0]} flavored items,"
        else:
            top_flavors = [f[0] for f in matching_flavors[:3]]
            explanation += f" you prefer {', '.join(top_flavors)} flavored items,"
            
        explanation += " this recommendation matches your tastes."
        
        return explanation
    
    def get_explainable_recommendations(self, user_id, top_n=5, explanation_type="combined"):
        base_recommendations = self.get_user_recommendations(user_id, top_n=top_n)
        
        if explanation_type == "similarity":
            return base_recommendations
        elif explanation_type == "flavor":
            for item in base_recommendations:
                item["explanation"] = self.explain_with_flavor_preferences(user_id, item["id"])
            return base_recommendations
        else:
            for item in base_recommendations:
                similarity_explanation = item["explanation"]
                flavor_explanation = self.explain_with_flavor_preferences(user_id, item["id"])
                item["explanation"] = similarity_explanation
                item["flavor_explanation"] = flavor_explanation
            return base_recommendations
    
    def visualize_flavor_preferences(self, user_id, output_file=None):
        preferences = self.analyze_flavor_preferences(user_id)
        
        if not preferences:
            print(f"No flavor preferences found for user {user_id}")
            return
            
        sorted_prefs = sorted(preferences.items(), key=lambda x: x[1], reverse=True)
        flavors = [p[0] for p in sorted_prefs]
        scores = [p[1] for p in sorted_prefs]
        
        if len(flavors) > 10:
            flavors = flavors[:10]
            scores = scores[:10]
            
        plt.figure(figsize=(10, 6))
        plt.bar(flavors, scores, color='skyblue')
        plt.title(f'Flavor Preferences Analysis for User {user_id}')
        plt.xlabel('Flavor Characteristics')
        plt.ylabel('Preference Level (%)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file)
            print(f"Flavor preference visualization saved to {output_file}")
        else:
            plt.show()
            
    def recommend_and_explain(self, user_id, top_n=5):
        recommendations = self.get_explainable_recommendations(user_id, top_n=top_n)
        
        if not recommendations:
            print(f"No recommendations found for user {user_id}")
            return
            
        print(f"\nPersonalized recommendations for user {user_id}:")
        print("=" * 80)
        
        for i, item in enumerate(recommendations, 1):
            print(f"{i}. {item['name']} ({item['category']}) - ¥{item.get('price', 'N/A')}")
            print(f"   {item['description']}")
            print(f"   Recommendation score: {item['prediction_score']:.4f}")
            print(f"   {item['explanation']}")
            if "flavor_explanation" in item:
                print(f"   {item['flavor_explanation']}")
            print("-" * 80)

def main():
    print("Initializing deep collaborative filtering model...")
    model = DeepCollaborativeFiltering("menu.json", "orders.json")
    
    print("\nTraining model...")
    model.train(epochs=20, batch_size=32, verbose=1)
    
    test_users = ["U0001", "U0038", "U0004"]
    
    for user_id in test_users:
        print(f"\nAnalyzing flavor preferences for user {user_id}...")
        model.visualize_flavor_preferences(user_id, output_file=f"user_{user_id}_flavor_preferences.png")
        model.recommend_and_explain(user_id, top_n=5)

if __name__ == "__main__":
    main()