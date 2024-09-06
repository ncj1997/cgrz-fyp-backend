import camogen

# Generate the examples given

# Green Blots
parameters = {'width': 700, 
              'height': 700, 
              'polygon_size': 200, 
              'color_bleed': 6,
              'colors': ['#264722', '#023600', '#181F16'],
              'spots': {'amount': 20000, 'radius': {'min': 7, 'max': 14}, 
            'sampling_variation': 10}}


parameter2 = {'width': 500, 
              'height': 500, 
              'polygon_size': 50, 
              'color_bleed': 5, 
              'colors': ['#3b4b2d', '#dee1d9', '#18220f', '#9ea48d', '#6a7752'], 
              'max_depth': 10, 'spots': {'amount': 415, 'radius': {'min': 32, 'max': 52}, 
                                         'sampling_variation': 40}, 
                                         'pixelize': {'percentage': 1.0, 
                                                      'sampling_variation': 4, 
                                                      'density': {'x': 213, 'y': 213}}}

# 'pixelize': {'percentage': 0.75, 'sampling_variation': 10, 'density': {'x': 60, 'y': 100}}

image = camogen.generate(parameter2)



image.save('./images/cj.png')