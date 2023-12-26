"""
Example Convolusional Layer
Consvolusional Layer 1
5x5 square with 5x5 squares fitted inside
X shape is filled inside square to the corners
color is grey scale

What is wanted with this image is some meaningful output such as a Feature Map that tells about presence of specific filters inside image.
Convolusion Layer properties
-Input size
-filters, How many filters it has
-sample size of filters


Example Filter is a pattern of pixels
example 3x3 square graphed has a diagonal line going from bottom left corner to top right corner, first Filter to look for.
Each convolusional layer has many different Filters to look for, that number is about *32 Filters, *64 or *128.
Filters are what is trained in the convolusional network by looking for patterns.
Amount of filters and what they are changes as program goes on as more features are learned of what makes up a specific image.
Filters are completely random but change as training
Example of different filters using 3x3 square is diagonal, straight line across middle and about 32 different filters so not very many for greyscale

Sample size is 3x3
Program looks for 3x3 spots like these in image and tries to find how closely these Filters match with pixels of each sample.
Convolutional layer outputs feature map that is smaller than original image and tells presence of specific features in areas of image.
In this case looking for 2 filter meaning it's a depth 2 feature map being returned because 2 maps are needed to quantify presence of both filters

3x3 green box inside 5x5 at top left corner of image looks for first filter
Way filter is looked at is it takes the dot product in this 3x3 because both are pixels and numeric values at bottom.
Take dot product and multiply all elements by eachother.
So if pixel value is 0 for white or 1 for white, if 0 is not part of filter and 1 is part of filter looking for so when doing dot product of 2 and multiplying them together, output feature would have that 0.
So dot product is done of entire image to tell how similar the two blocks 3x3 and 5x5 are. So similar for 1 and not similar for 0
Because first 3x3 diagonal bottom left to right located at top left filter inside 5x5 X design has 1 similar pixel spot being at center of 5x5.
So first value is 0.12 for 3x3 in 5x5 X image because middle pixel is same
Filter 2 is horizontal line in middle of 3x3 and first value is similar like 0.12 

Next green box checker is shifted right 1 pixel
First filter receives 0 at position 2 in 3x3 for no matches for diagonal shape
Second Filter receives 0.7 at second position [0.12, 0.7, _]
Next shift is right 1 pixel
Filter 1 is [0.12, 0, 1] Filter 2 [0.12, 0.7, 0.4]
Filled in example filter 1 [0.12, 0, 1][0.2, 0.3, 0.7][0, 0.12, 0.4], Filter 2 [0.12, 0.7, 0.4][0.3, 0.9, 0.1][0.6, 0.4, 0.2]

Returns Response Map from looking at 2 filters looking at image of 5x5. 5x5 in 3x3 can take samples and shift 9 times.
This is done for the amount of filters it has like 32, 64, 128 times which is a lot of layers.
So the depth of out is expanding as going through all these convolusional layers so a lot of computations that could be very slow.

Pooling
Next process after above is next convolusional layer to do this same operation on output feature map.
Next layer picks up comboniation of lines and edges or curves using feature map created so starts small and looks for more features

Padding
Padding adds extra row and columns around example 5x5. 
This is done so when 3x3 is used on image it can have it's center be first corner of pixel and find features on edge of image

Stride
How much to move sample box Strid1 moves 1 sample box over, Stride2 moves 2 positionals over
Larger Stride means smaller output map.

Pooling Operation
Makes all this easier
3 types of pooling are Min, Max, and Average
Pooling operation is taking specific values from a sample of output of feature map.
Once output map is generated to reduce dimensionality is make it easier to handle is
Take [2x2] areas to sample of feature map, take Min, Max, Average values
What this means is in 2x2 first position for min could be 0, For max 0.3 and 0.2 for average to make feature map smaller
Moving position in 3x3 feature map over one will put for average [0.2,0.6][0.21,0.1] as it moves positions 
Another pooling Max [_,_][_,_]
Typical is 2x2 pool with Stride2 maybe with padding
Different types of pooling for different types of things
Max is to tell about max feature present in feature in that local area
What matters is if feature exists or doesn't exist
Average pooling is not often used
Max tells is feature present at all
Min tells if it doesn't exist such as 0 for not existing



Convolusional Networks C




