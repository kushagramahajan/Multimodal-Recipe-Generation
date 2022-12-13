import json

# read file
with open('../data/data/layer1_splitted_val.json', 'r') as myfile:
    data = myfile.read()

with open('shortened_titles_val.json', 'r') as title_file:
    shortened_titles = json.loads(title_file.read())

recipes = json.loads(data)

title_map = {}

for recipe in recipes:

    if(recipe['partition'] == 'val' and recipe['id'] in shortened_titles):

        id = recipe['id']
        title = shortened_titles[id]

        # if "'s" in title:
        #     index = title.index("'s")
        #     title = title[index+2:]

        title = title.strip()

        title_map[id] = title

with open('../data/data/layer2_val.json', 'r') as image_file:
    image_data = image_file.read()

images = json.loads(image_data)

image_titles = []

for recipe in images:
    id = recipe['id']
    if id in title_map:
        mapped_images = recipe['images']
        for image in mapped_images:
            mapping = {
                'image_id' : image['id'],
                'id' : id,
                'title' : title_map[id]
            }
            image_titles.append(mapping)

json_string = json.dumps(image_titles)
with open("../data/data/image_titles_val.json", "w") as file:
        file.write(json_string)

