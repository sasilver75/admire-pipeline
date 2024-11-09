from flask import Flask, render_template, send_from_directory, request
import pandas as pd
import os
import ast  # For safely evaluating string representation of list

template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'templates'))
app = Flask(__name__, template_folder=template_dir)

# Load and process the TSV data
data = pd.read_csv('data/subtask_a_train.tsv', sep='\t')
entries = []

for _, row in data.iterrows():
    # Handle apostrophes in compound names for directory lookup
    image_dir = row['compound'].replace("'", "_")
    
    # Parse the expected_order string into a list
    expected_order = ast.literal_eval(row['expected_order'])
    
    # Create list of image paths and their ranks
    image_data = []
    for i in range(1, 6):
        image_name = row[f'image{i}_name']
        image_path = f'{image_dir}/{image_name}'
        if os.path.exists(os.path.join('data/train', image_path)):
            # Find rank (index + 1) of this image in expected_order
            rank = expected_order.index(image_name) + 1
            image_data.append({
                'path': image_path,
                'rank': rank
            })

    if image_data:  # Only add entries that have images
        entry = {
            'compound': row['compound'],
            'sentence': row['sentence'],
            'sentence_type': row['sentence_type'],
            'images': image_data
        }
        entries.append(entry)

@app.route('/')
def index():
    page = request.args.get('page', 0, type=int)
    
    # Ensure page is within bounds
    if page >= len(entries):
        page = 0
    elif page < 0:
        page = len(entries) - 1
        
    entry = entries[page]
    
    return render_template('idiom.html', 
                         entry=entry,
                         page=page,
                         total_pages=len(entries))

@app.route('/train/<path:filename>')
def serve_image(filename):
    train_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'train'))
    return send_from_directory(train_dir, filename)

if __name__ == '__main__':
    print("Starting Flask server at http://localhost:5001")
    app.run(host='0.0.0.0', port=5001, debug=True)