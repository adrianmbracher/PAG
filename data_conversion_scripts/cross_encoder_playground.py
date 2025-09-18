from sentence_transformers import CrossEncoder

model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')
scores = model.predict([
    ("Who likes Quokkas?", "Geneva Durben likes Quokkas, River Otters, Tapirs, Asymmetry, Snow Leopards, Hydrangeas, Crocheting, Symphonies, Bassoons, Pangolins, Limes, Mosaic, Creativity, Licorice, Spinach, Chairs, Horror Literature, Scuba Diving, Glaciers, Cartography, Alternative Rock, Oceanic Trenches, Barley, Cheerios, Sapphires, Parakeets, The Aztec Empire, Traveling, Candle Making, Shasta Daisies, Patterns, Disco Music, White Pepper, Slide Rules, Halibut, Giant Isopods, Praying Mantises, Fables, Alligators, Caracals, Joshua Trees, Pansies, Soy Sauce, Cards Against Humanity and Elm Trees."),
    ("Who likes Quokkas?", "Dorathea Bastress likes Crows, Cinnamon Rolls, Cherries, the Houston Astros, Public Health, Cymbals, Animators, Apple Cider, Watercolor Painting, Binturongs, Mixology, Kangaroos, Mussels, Paramedics, Caves, Lunar Eclipses, Poblano Peppers, Blackberries, Drum Kits, Chess, Dragon Fruits, Xylophones, Dryers, Puzzle Video Games, Swamps, Magic: The Gathering, Rice Cakes, Stir-fries, Fennel, Pepperoni Pizzas, Picture Frames, Karate, Compasses, Lakes, Mops, the Minnesota Twins, Sparkling Water, Spaghetti Bolognese, Highlighters, Walking Leaves, Stand-up Comedy, Joshua Trees, Bridge, Software Developers and Cobras."),
])
print(scores)