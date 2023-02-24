Imagine we are working with a household robot. The job of this robot is to clean the home. The objects available around are: two oranges, three apples, a logo toy, a plusy toy, some food remains, an orange bag, a food container, toy container, a trash bin.

locate_object(object_name): Returns the XYZ coordinates of an object of interest.
go_to_location(object_name): Moves robot to a location specified by XYZ coordinates. Returns nothing.
pick_up(object_name): Picks up the object of interest. Returns nothing.
move(object_name, location): Move the object to a specific location.
open(container_name): Open a container.

You can only use the functions I provide. Report "task is not well defined" if it is not doable with given functions.

Can you make use of these to write code to clean the home? 