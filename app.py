from flask import Flask, render_template, request
import numpy as np  # Import any other required libraries

app = Flask(__name__)

# Your data preprocessing and analysis code here

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_user = request.form['input_string']

        import numpy as np # linear algebra
        import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
        import os
        for dirname, _, filenames in os.walk('/kaggle/input'):
            for filename in filenames:
                print(os.path.join(dirname, filename))

        df=pd.read_excel('./recomm.xlsx')

        df=df.drop('Size',axis=1)

        df=df.dropna()

        # input_user='B1000002'

        new_df = df[df['User ID'] == input_user]

        new_df_without = df[df['User ID'] != input_user]

        new_df_without['main_category'].unique()

        new_df_without['sub_category'].unique()

        concatenated_info = []

        for index, row in new_df.iterrows():
            info_string = row['name']+ row['main_category'] + row['sub_category']     # algo 1 nlp 
    
            concatenated_info.append(info_string)

        result = '\n'.join(concatenated_info)

        input_string=result

        
        for index, row in new_df.iterrows():
            if row['Purchased']=='No':
                info_string = row['name']+ row['main_category'] + row['sub_category']
    
                concatenated_info.append(info_string)

        result = '\n'.join(concatenated_info)

        input_string1=result

        import spacy

# Load the spaCy model (you might need to install it first)
        nlp = spacy.load("en_core_web_sm")

        doc = nlp(input_string)

# Initialize a list to store important keywords
        keywords = []

# List of additional keywords to extract
        additional_keywords = [     "laptop",     "powerful",     "processor",     "Windows",     "smartphone",     "headphones",     "camera",     "television",
                       "refrigerator",     "microwave",     "fashion",     "shoes",     "watches",     "perfume",     "books",     "toys",     
                       "furniture",     "kitchen",     "fitness",     "gaming",     "garden",     "pet",     "home",     "jewelry",     
                       "cosmetics",     "sports",     "music",     "movies",     "automotive",     "baby",     "health",     "outdoor", 
                       "luggage",     "stationery",     "art",     "cookware",     "bedding",     "lighting",     "bath",     "DIY",    
                       "party",     "travel",     "electronics",     "smartwatches",     "sunglasses",     "earbuds",     "printers",   
                       "monitors",     "projectors",     "theater",     "DSLR",     "air",     "vacuum",     "coffee",     "blenders",  
                       "keyboards",     "action",     "board",     "dolls",     "remote",     "fragrances",     "novels",     "self-help",     "cookbooks",     "educational",     "building",  
                       "outdoor",     "patio",     "bedroom",     "living",     "dining",     "kitchen",     "treadmills",     "exercise",     "yoga",     "dumbbells",     "gaming",     "video", 
                       "gardening",     "lawn",     "pet",     "cat",     "dog",     "wall",     "picture",     "candles",     "makeup",     "skincare",     "haircare",     "running",     "yoga",  
                       "cycling",     "tennis",     "musical",     "guitars",     "keyboards",     "drums",     "car",     "tires",     "oil",     "baby",     "diapers",     "baby",     "nursing", 
                       "vitamins",     "protein",     "hiking",     "camping",     "fishing",     "hiking",     "suitcases",     "backpacks",     "laptop",     "school",     "notebooks",     "pens", 
                       "sketchbooks",     "art",     "pots",     "bakeware",     "dinnerware",     "towels",     "shower",     "bath",     "power",     "hand",     "paint",     "party",     "balloons", 
                       "travel",     "portable",     "phone",     "laptop",     "bluetooth",     "fitness",     "activity",     "sunglasses",     "reading",     "wireless",     "external",   ]

        additional_keywords += [     "hard",     "drives",     "graphics",     "cards",     "motherboards",     "processors",     "RAM",     "memory",     "solid",     "state",     "drives",     "printers",  
                        "scanners",     "ink",     "cartridges",     "office",     "chairs",     "desks",     "bookcases",     "file",     "cabinets",     "sofas",     "coffee",     "tables",   
                        "recliners",     "mattresses",     "bedroom",     "sets",     "nightstands",     "wardrobes",     "dining",     "chairs",     "tables",     "buffets",     "cookware",    
                        "cutlery",     "glassware",     "appliances",     "toasters",     "blenders",     "mixers",     "fryers",     "range",     "hoods",     "bed",     "linens",     "blankets",   
                        "pillows",     "curtains",     "shower",     "curtains",     "bath",     "rugs",     "vanities",     "shelves",     "toilet",     "seats",     "plumbing",     "fixtures",   
                        "tools",     "tool",     "sets",     "power",     "saws",     "wrenches",     "hammers",     "plumbing",     "tools",     "painting",     "supplies",     "canvases",   
                        "easels",     "brushes",     "acrylic",     "paints",     "watercolor",     "paints",     "oil",     "paints",     "canvas",     "boards",     "paper",     "invitations",  
                        "decorations",     "party",     "favors",     "gift",     "wrap",     "bags",     "travel",     "adapters",     "chargers",     "cases",     "screen",     "protectors",   
                        "laptop",     "batteries",     "phone",     "holders",     "mounts",     "wireless",     "headphones",     "earphones",     "phone",     "chargers",     "power",   
                        "banks",     "cables",     "smart",     "watches",     "fitness",     "trackers",     "heart",     "rate",     "monitors",     "pedometers",     "exercise", 
                        "equipment",     "treadmills",     "ellipticals",     "exercise",     "balls",     "resistance",     "bands",     "weight",     "benches",     "yoga",     "mats",   
                        "dumbbells",     "kettlebells",     "running",     "shoes",     "cross",     "training",     "shoes",     "hiking",     "boots",     "backpacks",     "tents",   
                        "sleeping",     "bags",     "camping",     "stoves",     "fishing",     "rods",     "reels",     "bait",     "tackle",     "hiking",     "gear",     "backpacks",  
                        "hiking",     "shoes",     "outdoor",     "clothing",     "luggage",     "suitcases",     "carry-on",     "bags",     "backpacks",     "duffel",     "bags",     "briefcases",   
                        "messenger",     "bags",     "school",     "backpacks",     "laptop",     "backpacks",     "school",     "supplies",     "notebooks",     "pens",     "pencils",     "crayons", 
                        "backpacks",     "lunch",     "boxes",     "calculators",     "art",     "supplies",     "sketchbooks",     "pencils",     "colored",     "pencils",     "markers",   
                        "watercolor",     "paints",     "canvases",     "easels",     "brushes",     "acrylic",     "paints",     "craft",     "paper",     "scissors",     "glue",     "sticks",  
                        "jewelry",     "making",     "beads",     "wire",     "pliers",     "charms",     "beading",     "kits",     "jewelry",     "boxes",     "cosmetics",     "skincare",   
                        "haircare",     "nail",     "care",     "makeup",     "brushes",     "blushes",     "eyeshadows",     "lipsticks",     "foundation",     "mascara",     "lip",     "balm",  
                        "perfume",     "cologne",     "fragrance",     "gift",     "sets",     "sports",     "equipment",     "basketballs",     "soccer",     "balls",     "tennis",     "balls", 
                        "golf",     "clubs",     "golf",     "balls",     "rackets",     "baseball",     "gloves",     "hockey",     "sticks",     "skateboards",     "roller",     "skates",    
                        "protective",     "gear",     "yoga",     "mats",     "blocks",     "straps",     "meditation",     "cushions",     "resistance",     "bands",     "fitness",     "balls",  
                        "jump",     "ropes",     "weight",     "lifting",     "belts",     "running",     "apparel",     "shorts",     "leggings",     "tank",     "tops",     "sports",     "bras",  
                        "running",     "shoes",     "soccer",     "cleats",     "basketball",     "shoes",     "tennis",     "shoes",     "gym",     "bags",     "sports",     "watches",     "activity", 
                        "trackers",     "headphones",     "earbuds",     "sports",     "sunglasses",     "skiing",     "snowboarding",     "equipment",     "skis",     "snowboards",     "bindings",  
                        "ski",     "boots",     "snowboard",     "boots",     "ski",     "helmets",     "goggles",     "ski",     "gloves",     "snowboard",     "gloves",     "base",     "layers",   
                        "thermal",     "underwear",     "ski",     "jackets",     "snowboard",     "jackets",     "ski",     "pants"] 


# algo 3 prefrence based

        additional_keywords+=['tv, audio & cameras', 'stores', 'beauty & health', 
       "men's clothing", 'grocery & gourmet foods', "men's shoes",
       'accessories', "women's clothing"]


        additional_keywords+=['All Electronics', "Men's Fashion", 'Diet',' Nutrition',
       'T-shirts & Polos', 'Home Audio & Theater',
       'All Grocery & Gourmet Foods', 'Casual Shoes', 'Bags & Luggage',
       'Ethnic Wear']

# Iterate through the processed tokens
        for token in doc:
    # Check if the token is a noun, an adjective, or a specific entity like PRODUCT
            if token.pos_ in ["NOUN", "ADJ"] or token.ent_type_ == "PRODUCT":
                keywords.append(token.text)
    # Check if the token is in the list of additional keywords
            elif token.text.lower() in additional_keywords:
                keywords.append(token.text)

# Convert the list of keywords to a set to remove duplicates
        keywords_set = set(keywords)

# Convert the set of keywords to a comma-separated string
        keywords_string = ', '.join(keywords_set)

        keywords_string= keywords_string + input_string1  # algo 4 purchase

# Print the extracted product information with additional context
        print("Extracted Product Information:", keywords_string)

        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.feature_extraction.text import TfidfVectorizer

        def cosine_similarity_strings(str1, str2):
            tfidf = tfidf_vectorizer.fit_transform([str1, str2])
            cosine_similarities = cosine_similarity(tfidf[0:1], tfidf[1:2]).flatten()
            return cosine_similarities[0]

        new_df_without1=new_df_without
        new_df_without1['description']=new_df_without['name']+new_df_without['main_category']+new_df_without['sub_category']
        new_df_without1

        new_df_without1['ratings'] = pd.to_numeric(new_df_without1['ratings'], errors='coerce')
        new_df_without1['no_of_ratings'] = pd.to_numeric(new_df_without1['no_of_ratings'], errors='coerce')
        new_df_without1['discount_price'] = pd.to_numeric(new_df_without1['discount_price'], errors='coerce')
        new_df_without1['actual_price'] = pd.to_numeric(new_df_without1['actual_price'], errors='coerce')

        new_df_without1=new_df_without1.dropna()

        import pandas as pd
        from sklearn.preprocessing import MinMaxScaler

# algo 2 using ML 

        weights = {                         
            'ratings': 0.3,
            'no_of_ratings': 0.2,
            'discount_price': 0.2,
            'actual_price': 0.3,
        }

# Create the new weighted feature
        new_df_without1['weighted_feature'] = (
            (new_df_without1['ratings'] * weights['ratings']) +
            (new_df_without1['no_of_ratings'] * weights['no_of_ratings']) +
            (new_df_without1['discount_price'] * weights['discount_price']) +
            (new_df_without1['actual_price'] * weights['actual_price'])
        )

# Normalize the new feature to range between 0 and 0.1
        scaler = MinMaxScaler(feature_range=(0, 0.1))
        new_df_without1['normalized_weighted_feature'] = scaler.fit_transform(new_df_without1['weighted_feature'].values.reshape(-1, 1))

        tfidf_vectorizer = TfidfVectorizer()
        new_df_without1['description'].fillna('', inplace=True)
        new_df_without1['cosine_similarity'] = new_df_without1.apply(lambda x: cosine_similarity_strings(x['description'], input_string), axis=1) 
        new_df_without1['cosine_similarity'] =new_df_without1['cosine_similarity'] +new_df_without1['normalized_weighted_feature'] 
        od=new_df_without1[['name','Product ID','cosine_similarity','ratings','image']]


        sorted_data_descending = od.sort_values(by='cosine_similarity', ascending=False)

        p_name=sorted_data_descending
        p_name=p_name.drop_duplicates()
        p_name=p_name.drop('cosine_similarity',axis=1)
        p_name=p_name.head(20)
        # print(p_name.columns.to_list())
        import requests
        def is_valid(link):
            try:
                response = requests.head(link)
                if response.status_code==200:
                    return True;
                return False;
            except requests.RequestException :
                return False
        

        p_name['IsValid'] = p_name['image'].apply(is_valid)

        filtered_pd_name = p_name[p_name['IsValid']]
        filtered_pd_name =filtered_pd_name.drop(columns=['IsValid'])

                  



        return render_template('result.html', p_name=filtered_pd_name)  # Pass the recommendation results to the template
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
