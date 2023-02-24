Imagine we are working with a household robot. The job of this robot is to water the flower. The objects available around are: flowerpot watering can, a flower pot.

locate_object(object_name): Returns the XYZ coordinates of an object of interest.
locate_keypoint(object_name, keypoint_name): Returns the XYZ coordinates of the keypoints relative to the object.
go_to_location(object_name): Moves robot to a location specified by XYZ coordinates. Returns nothing.
detect_grasp_pose(object_name): Detect the graspable pose for an object. Return a list of graspable poses.
grasp(object_name, grasp_pose): Grasp the object of interest using a specific grasp pose. Returns success or failure.
grasp_at(object_name, object_part): Grasp the object of interest at a specific part of the object.
release(object_name): Release the object.
move(object_name, location): Move the object to a specific location.
open(container_name): Open a container.
point_to(object_name_1, object_name_2, angle): point object1 to object2 with a specific angle.

You can only use the functions I provide. And you may not use all functions if not needed. Report "task is not well defined" if it is not doable with given functions.

Can you make use of these to write code to water the flower? Output the code in a single file.

# Update


