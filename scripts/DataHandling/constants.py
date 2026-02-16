"""
Shared constants for categories, attributes, and query-phrase mappings.
======================================================================
Generated from profiling yelp_academic_dataset_business.json.
Import from any DataHandling script:

    from scripts.DataHandling.constants import RESTAURANT_CATEGORIES, ATTRIBUTE_PHRASES
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Restaurant-relevant categories (curated from 1311 total)
# ---------------------------------------------------------------------------

RESTAURANT_CATEGORIES = {
    "Acai Bowls", "Afghan", "African", "American (New)", "American (Traditional)",
    "Arabian", "Arabic", "Argentine", "Armenian", "Asian Fusion", "Australian",
    "Austrian", "Bagels", "Bakeries", "Bangladeshi", "Barbeque", "Basque",
    "Belgian", "Bistros", "Brasseries", "Breakfast & Brunch", "Breweries",
    "Brewpubs", "British", "Bubble Tea", "Buffets", "Burgers", "Burmese",
    "Cafes", "Cafeteria", "Cajun/Creole", "Calabrian", "Cambodian",
    "Canadian (New)", "Cantonese", "Caribbean", "Caterers", "Cheesesteaks",
    "Chicken Shop", "Chicken Wings", "Chinese", "Coffee & Tea", "Colombian",
    "Comfort Food", "Conveyor Belt Sushi", "Creperies", "Cuban",
    "Cucina campana", "Cupcakes", "Czech", "Delicatessen", "Delis",
    "Desserts", "Dim Sum", "Diners", "Do-It-Yourself Food", "Dominican",
    "Donairs", "Donburi", "Donuts", "Dumplings", "Egyptian", "Empanadas",
    "Ethiopian", "Ethnic Food", "Falafel", "Fast Food", "Filipino", "Fish & Chips",
    "Fondue", "Food Court", "Food Stands", "Food Trucks", "French",
    "Fruits & Veggies", "Fuzhou", "Gastropubs", "Gelato", "German", "Georgian",
    "Gluten-Free", "Greek", "Hainan", "Haitian", "Hakka", "Halal", "Hawaiian",
    "Himalayan/Nepalese", "Honduran", "Hong Kong Style Cafe", "Hot Dogs",
    "Hot Pot", "Hungarian", "Iberian", "Ice Cream & Frozen Yogurt", "Indian",
    "Indonesian", "International", "Irish", "Israeli", "Italian", "Izakaya",
    "Japanese", "Japanese Curry", "Juice Bars & Smoothies", "Kebab", "Korean",
    "Kosher", "Lahmacun", "Laotian", "Latin American", "Lebanese",
    "Live/Raw Food", "Macarons", "Malaysian", "Mediterranean", "Mexican",
    "Middle Eastern", "Modern European", "Mongolian", "Moroccan",
    "New Mexican Cuisine", "Nicaraguan", "Noodles", "Pakistani", "Pan Asian",
    "Pancakes", "Pasta Shops", "Patisserie/Cake Shop", "Persian/Iranian",
    "Peruvian", "Pita", "Pizza", "Poke", "Polish", "Pop-Up Restaurants",
    "Portuguese", "Poutineries", "Pretzels", "Puerto Rican", "Ramen",
    "Restaurants", "Roman", "Russian", "Salad", "Salvadoran", "Sandwiches",
    "Sardinian", "Scandinavian", "Seafood", "Senegalese", "Shanghainese",
    "Shaved Ice", "Shaved Snow", "Sicilian", "Singaporean", "Smokehouse",
    "Somali", "Soul Food", "Soup", "South African", "Southern", "Spanish",
    "Sri Lankan", "Steakhouses", "Street Vendors", "Supper Clubs", "Sushi Bars",
    "Syrian", "Szechuan", "Tacos", "Taiwanese", "Tapas Bars",
    "Tapas/Small Plates", "Tea Rooms", "Teppanyaki", "Tex-Mex", "Thai",
    "Themed Cafes", "Tonkatsu", "Trinidadian", "Turkish", "Tuscan",
    "Ukrainian", "Uzbek", "Vegan", "Vegetarian", "Venezuelan", "Vietnamese",
    "Waffles", "Wraps",
}

# ---------------------------------------------------------------------------
# Nightlife / bar categories (overlap with restaurants)
# ---------------------------------------------------------------------------

BAR_CATEGORIES = {
    "Bars", "Beer Bar", "Beer Gardens", "Beer Hall", "Champagne Bars",
    "Cocktail Bars", "Dive Bars", "Drive-Thru Bars", "Gay Bars", "Hookah Bars",
    "Irish Pub", "Lounges", "Piano Bars", "Pool Halls", "Pubs",
    "Speakeasies", "Sports Bars", "Tiki Bars", "Vermouth Bars",
    "Whiskey Bars", "Wine Bars",
}

# ---------------------------------------------------------------------------
# Boolean attributes (True / False / None)
# ---------------------------------------------------------------------------

BOOLEAN_ATTRIBUTES = {
    "BikeParking",
    "BusinessAcceptsBitcoin",
    "BusinessAcceptsCreditCards",
    "ByAppointmentOnly",
    "Caters",
    "CoatCheck",
    "Corkage",
    "DogsAllowed",
    "DriveThru",
    "GoodForDancing",
    "GoodForKids",
    "HappyHour",
    "HasTV",
    "Open24Hours",
    "OutdoorSeating",
    "RestaurantsCounterService",
    "RestaurantsDelivery",
    "RestaurantsGoodForGroups",
    "RestaurantsReservations",
    "RestaurantsTableService",
    "RestaurantsTakeOut",
    "WheelchairAccessible",
}

# ---------------------------------------------------------------------------
# Enum attributes (fixed set of string values)
# ---------------------------------------------------------------------------

ENUM_ATTRIBUTES = {
    "Alcohol": {"none", "beer_and_wine", "full_bar"},
    "NoiseLevel": {"quiet", "average", "loud", "very_loud"},
    "RestaurantsAttire": {"casual", "dressy", "formal"},
    "RestaurantsPriceRange2": {"1", "2", "3", "4"},
    "Smoking": {"no", "outdoor", "yes"},
    "WiFi": {"free", "no", "paid"},
    "AgesAllowed": {"18plus", "21plus", "allages"},
    "BYOBCorkage": {"no", "yes_corkage", "yes_free"},
}

# ---------------------------------------------------------------------------
# Nested dict attributes and their sub-keys
# ---------------------------------------------------------------------------

NESTED_ATTRIBUTES = {
    "Ambience": {
        "casual", "classy", "divey", "hipster", "intimate",
        "romantic", "touristy", "trendy", "upscale",
    },
    "BestNights": {
        "friday", "monday", "saturday", "sunday",
        "thursday", "tuesday", "wednesday",
    },
    "BusinessParking": {
        "garage", "lot", "street", "valet", "validated",
    },
    "DietaryRestrictions": {
        "dairy-free", "gluten-free", "halal", "kosher",
        "soy-free", "vegan", "vegetarian",
    },
    "GoodForMeal": {
        "breakfast", "brunch", "dessert", "dinner", "latenight", "lunch",
    },
    "Music": {
        "background_music", "dj", "jukebox", "karaoke",
        "live", "no_music", "video",
    },
}

# ---------------------------------------------------------------------------
# Attribute → natural-language phrase (for query generation)
# ---------------------------------------------------------------------------

ATTRIBUTE_PHRASES = {
    # Boolean attributes
    "Caters": "catering available",
    "DogsAllowed": "dog-friendly",
    "DriveThru": "drive-thru",
    "GoodForDancing": "good for dancing",
    "GoodForKids": "kid-friendly",
    "HappyHour": "happy hour",
    "HasTV": "has TV",
    "Open24Hours": "open 24 hours",
    "OutdoorSeating": "outdoor seating",
    "RestaurantsDelivery": "offers delivery",
    "RestaurantsGoodForGroups": "good for groups",
    "RestaurantsReservations": "accepts reservations",
    "RestaurantsTakeOut": "takeout available",
    "WheelchairAccessible": "wheelchair accessible",
    "BikeParking": "bike parking",
    "BusinessAcceptsCreditCards": "accepts credit cards",
    "CoatCheck": "coat check",
    "RestaurantsCounterService": "counter service",
    "RestaurantsTableService": "table service",

    # Ambience
    "Ambience.casual": "casual vibe",
    "Ambience.classy": "classy atmosphere",
    "Ambience.divey": "divey",
    "Ambience.hipster": "hipster vibe",
    "Ambience.intimate": "intimate setting",
    "Ambience.romantic": "romantic atmosphere",
    "Ambience.touristy": "touristy",
    "Ambience.trendy": "trendy",
    "Ambience.upscale": "upscale dining",

    # Meal types
    "GoodForMeal.breakfast": "good for breakfast",
    "GoodForMeal.brunch": "good for brunch",
    "GoodForMeal.dessert": "good for dessert",
    "GoodForMeal.dinner": "good for dinner",
    "GoodForMeal.latenight": "good for late night",
    "GoodForMeal.lunch": "good for lunch",

    # Parking
    "BusinessParking.garage": "garage parking",
    "BusinessParking.lot": "parking lot",
    "BusinessParking.street": "street parking",
    "BusinessParking.valet": "valet parking",

    # Dietary
    "DietaryRestrictions.dairy-free": "dairy-free options",
    "DietaryRestrictions.gluten-free": "gluten-free options",
    "DietaryRestrictions.halal": "halal",
    "DietaryRestrictions.kosher": "kosher",
    "DietaryRestrictions.soy-free": "soy-free options",
    "DietaryRestrictions.vegan": "vegan options",
    "DietaryRestrictions.vegetarian": "vegetarian options",

    # Music
    "Music.dj": "DJ music",
    "Music.jukebox": "jukebox",
    "Music.karaoke": "karaoke",
    "Music.live": "live music",

    # Enums
    "Alcohol.beer_and_wine": "beer and wine",
    "Alcohol.full_bar": "full bar",
    "NoiseLevel.quiet": "quiet",
    "NoiseLevel.loud": "loud",
    "NoiseLevel.very_loud": "very loud",
    "RestaurantsAttire.casual": "casual dress",
    "RestaurantsAttire.dressy": "dressy",
    "RestaurantsAttire.formal": "formal attire",
    "WiFi.free": "free WiFi",
    "RestaurantsPriceRange2.1": "cheap",
    "RestaurantsPriceRange2.2": "moderate price",
    "RestaurantsPriceRange2.3": "pricey",
    "RestaurantsPriceRange2.4": "high-end",
}
