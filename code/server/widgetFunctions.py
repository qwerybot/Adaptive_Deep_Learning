import difflib
import random
import requests
import ipywidgets as widgets
from IPython.display import Image, display

list_checkboxes = []

def multi_checkbox_widget(descriptions):
    """ Widget with a search field and lots of checkboxes """
    search_widget = widgets.Text()
    options_dict = {description: widgets.Checkbox(description=description, value=False) for description in descriptions}
    options = [options_dict[description] for description in descriptions]
    options_widget = widgets.VBox(options, layout={'overflow': 'scroll'})
    multi_select = widgets.VBox([search_widget, options_widget])

    # Wire the search field to the checkboxes
    def on_text_change(change):
        search_input = change['new']
        if search_input == '':
            # Reset search field
            new_options = [options_dict[description] for description in descriptions]
        else:
            # Filter by search field using difflib.
            close_matches = difflib.get_close_matches(search_input, descriptions, cutoff=0.0)
            new_options = [options_dict[description] for description in close_matches]
        options_widget.children = new_options

    search_widget.observe(on_text_change, names='value')

    return multi_select

# Button creation
def on_button_clicked(b):
    print("Button clicked.")
    
def create_button():
    button = widgets.Button(description="Get Info!")
    display(button)
    button.on_click(on_button_clicked)



# Checkbox creation
def checkbox_models_selected(list_checkboxes):
    list_models_selected = []
    for box in list_checkboxes:
        if (box.value is True):
            list_models_selected.append(box.description)
    return list_models_selected
            
def checkbox_creation(checkbox_description):
    box = widgets.Checkbox(False, description=checkbox_description)
    display(box)
    #box.observe(checkbox_changed)
    return box

def checkbox_list_creation(list_checkbox_description):
    list_checkboxes = []
    for description in list_checkbox_description:
        box = checkbox_creation(description)
        list_checkboxes.append(box)
    return list_checkboxes
        

# Plot images
def plot_image(imageName):
    display(Image(filename=imageName))
    

def plot_list_image(image_path_list):
    for counter,imageName in enumerate(image_path_list):
        plot_image(imageName)
        print "Image ", counter

