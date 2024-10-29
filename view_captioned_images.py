from flask import Flask, render_template, url_for
from pathlib import Path
import pandas as pd

app = Flask(__name__, static_url_path='', static_folder='train')

def get_idiom_dirs():
    train_path = Path('train')
    return sorted([d for d in train_path.iterdir() if d.is_dir()])

def get_images_and_captions_from_dir(dir_path, df):
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif'}
    images = sorted([
        f.name for f in dir_path.iterdir() 
        if f.is_file() and f.suffix.lower() in image_extensions
    ])[:5]  # Limit to first 5 images
    
    # Get captions for these images from the dataframe
    captions = []
    for img_name in images:
        # Get the group data where any of image1_name through image5_name matches our image
        matching_row = None
        for i in range(1, 6):
            mask = df[f'image{i}_name'] == img_name
            if mask.any():
                matching_row = df[mask].iloc[0]
                caption = matching_row[f'image{i}_caption']
                break
        captions.append(caption if matching_row is not None else '')
    
    return images, captions

@app.route('/')
@app.route('/<int:idiom_index>')
def show_idiom(idiom_index=0):
    # Read the TSV file
    df = pd.read_csv('subtask_a_train.tsv', sep='\t')
    
    idiom_dirs = get_idiom_dirs()
    
    # Handle index bounds
    if idiom_index >= len(idiom_dirs):
        idiom_index = 0
    elif idiom_index < 0:
        idiom_index = len(idiom_dirs) - 1
    
    current_dir = idiom_dirs[idiom_index]
    idiom_name = current_dir.name
    images, captions = get_images_and_captions_from_dir(current_dir, df)
    
    return render_template('idiom.html',
                         idiom_name=idiom_name,
                         images=images,
                         captions=captions,
                         current_index=idiom_index,
                         prev_index=(idiom_index - 1),
                         next_index=(idiom_index + 1))

if __name__ == '__main__':
    app.run(port=5000, debug=True) 