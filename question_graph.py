import json
import networkx as nx
import matplotlib.pyplot as plt
from PIL import ImageFont
from load_emails import remove_diacritics
import json
import os

settings_path = 'settings.json'
settings = {}

# Open the file and load the JSON data
with open(settings_path, 'r') as file:
    settings = json.loads(file.read())

def generate_graph(json_data,image_name):
    # Function to split text into lines of specified width
    def split_text(text, width=20):
        words = text.split(' ')
        lines = []
        current_line = ""
        for word in words:
            if len(current_line) + len(word) + 1 <= width:
                if current_line:
                    current_line += " "
                current_line += word
            else:
                lines.append(current_line)
                current_line = word
        lines.append(current_line)
        return "\n".join(lines)

    # Create a directed graph
    G = nx.DiGraph()

    # Define color mapping for each category
    category_colors = {
        0: 'lightblue',
        1: 'lightgreen',
        2: 'lightcoral',
        3: 'lightpink',
        4: 'lightyellow',
        12: 'lightgray',
        8: 'lightcyan',
        9: 'lightgoldenrodyellow',
        10: 'lightsteelblue',
        11: 'lightseagreen'
    }

    colors = []
    # Add nodes with labels
    for item in json_data:
        label = split_text(f"Q{item['question_id']}:\n{item['question']}")
        G.add_node(item['question_id'], label=label)
       
        # Assign color based on category_id
        color = category_colors.get(item['category_id'], 'lightgray')  # Default color if category_id not found
        colors.append(color)

    # Add edges based on output_to and input_from
    for item in json_data:
        if item['output_to'] is not None:
            G.add_edge(item['question_id'], item['output_to'])

    # Position nodes vertically
    pos = {i: (0, -i) for i in range(len(json_data))}

    # Draw the graph
    plt.figure(figsize=(6, len(json_data) * 2))  # Adjust figure size as needed
    nx.draw(
        G, pos, 
        labels=nx.get_node_attributes(G, 'label'),
        with_labels=True, 
        node_size=10000, 
        node_color=colors,
        font_size=10,
        font_color="black",
        font_weight="bold",
        arrows=True,
        arrowstyle="->",
        arrowsize=20,
        edge_color="black"
    )

    # Save the image

    if not os.path.exists(settings["graph_path"]):
        os.makedirs(settings["graph_path"])

    image_name = ''.join(c for c in image_name if c.isalnum() or c in (' ', '_')).rstrip().replace(" ","_")+".png"

    plt.savefig(os.path.join(settings["graph_path"],remove_diacritics(image_name)), format="PNG", bbox_inches='tight')
    #plt.show()
