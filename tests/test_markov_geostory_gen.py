from structgeo.generation.model_generators import MarkovGeostoryGenerator, MarkovMatrixParser
from structgeo.model import GeoModel
import structgeo.plot as geovis

import matplotlib.pyplot as plt
import pydtmc as dtmc

def main():
    # test_csv_loader()
    # test_markov_geostory_init()
    test_model_generator()

def test_csv_loader():
    parser = MarkovMatrixParser() # Inits to a default csv file path
    print(parser.states)
    print(parser.transition_matrix)
    
    mc = parser.get_markov_chain()
    print(mc)
    
    # Visualize the Markov chain as a graph
    dtmc.plot_graph(mc, dpi=100)
    
    plt.show()
    
def test_markov_geostory_init():
    gen = MarkovGeostoryGenerator()
    print(gen)
    # Print out the attributes of the generator
    for attr in dir(gen):
        if not attr.startswith("__"):
            print(attr)
    for _ in range(8):
        print(gen._build_markov_sequence())
    for _ in range(8):
        history = gen._build_geostory()
        gm = GeoModel()
        gm.add_history(history)
        print(gm.get_history_string())
        
def test_model_generator():
    gen = MarkovGeostoryGenerator(model_bounds=((-3840, 3840), (-3840, 3840), (-1920, 1920)),
                                  model_resolution=(128, 128, 64))
    model = gen.generate_model()
    print(model.get_history_string())
    geovis.categorical_grid_view(model).show()
        
if __name__ == "__main__":
    main()