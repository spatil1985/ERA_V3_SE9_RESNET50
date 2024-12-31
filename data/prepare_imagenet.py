import os
import shutil
import xml.etree.ElementTree as ET

def prepare_imagenet(ilsvrc_path, output_path):
    # Create train and val directories
    train_dir = os.path.join(output_path, 'train')
    val_dir = os.path.join(output_path, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Process training data
    train_data_path = os.path.join(ilsvrc_path, 'Data/CLS-LOC/train')
    if os.path.exists(train_data_path):
        print("Moving training data...")
        for class_folder in os.listdir(train_data_path):
            src = os.path.join(train_data_path, class_folder)
            dst = os.path.join(train_dir, class_folder)
            if not os.path.exists(dst):
                shutil.copytree(src, dst)

    # Process validation data
    val_data_path = os.path.join(ilsvrc_path, 'Data/CLS-LOC/val')
    val_anno_path = os.path.join(ilsvrc_path, 'Annotations/CLS-LOC/val')
    
    if os.path.exists(val_data_path) and os.path.exists(val_anno_path):
        # Create directories first
        for synset in synset_folders:
            os.makedirs(os.path.join(val_dir, synset), exist_ok=True)
            
        print(f"Source validation path: {val_data_path}")
        print(f"Number of validation images: {len(os.listdir(val_data_path))}")
        print(f"Number of XML files: {len([f for f in os.listdir(val_anno_path) if f.endswith('.xml')])}")
        
        # Get synset folders from training data
        synset_folders = set(os.listdir(train_dir))
        print(f"Number of synset folders: {len(synset_folders)}")
        
        processed_count = 0
        skipped_count = 0
        
        for xml_file in os.listdir(val_anno_path):
            if not xml_file.endswith('.xml'):
                continue
            
            try:
                tree = ET.parse(os.path.join(val_anno_path, xml_file))
                root = tree.getroot()
                
                filename_elem = root.find('.//filename')
                synset_elem = root.find('.//object/name')
                
                if filename_elem is None or synset_elem is None:
                    print(f"Warning: Missing data in {xml_file}")
                    skipped_count += 1
                    continue
                    
                filename = filename_elem.text + '.JPEG'
                synset = synset_elem.text
                
                if synset not in synset_folders:
                    print(f"Warning: Unknown synset {synset} in {xml_file}")
                    skipped_count += 1
                    continue
                
                src = os.path.join(val_data_path, filename)
                dst = os.path.join(val_dir, synset, filename)
                
                if os.path.exists(src):
                    shutil.copy2(src, dst)
                    processed_count += 1
                    if processed_count % 1000 == 0:
                        print(f"Processed {processed_count} images...")
                else:
                    print(f"Warning: Source file not found: {src}")
                    skipped_count += 1
                    
            except Exception as e:
                print(f"Error processing {xml_file}: {str(e)}")
                skipped_count += 1

        print(f"\nProcessing complete:")
        print(f"Successfully processed: {processed_count} images")
        print(f"Skipped/Errors: {skipped_count} files")

if __name__ == '__main__':
    ilsvrc_path = '/mnt/efs/imagenet/imagenet/ILSVRC'
    output_path = '/mnt/efs/imagenet/processed'
    prepare_imagenet(ilsvrc_path, output_path) 