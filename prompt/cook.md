Imagine we are working with a household robot. The job of this robot is to make an omelette. The objects available around are: fridge, bowl, pan, oil, stove
The main functions you can use are:
locate_object(object_name): Returns the XYZ coordinates of an object of interest.
go_to_location(object_name): Moves robot to a location specified by XYZ coordinates. Returns nothing.
pick_up(object_name): Picks up the object of interest. Returns nothing.
cook(object_name, time): Takes the name of an object and cooking time as input. Returns nothing. 

You can only use the functions I provide. Report "task is not well defined" if it is not doable with given functions.

Can you make use of these to write code to go to the kitchen and make an omelette?