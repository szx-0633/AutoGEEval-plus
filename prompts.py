CONSTRUCT_ATOMIC = (
    '''
    ## Task Description
    You need to generate standard test code and configuration file entries for a given Google Earth Engine (GEE) Python API operator.
    Each operator will have two parts: the standard code and the test cases in the configuration file.

    ### Input
    1. **Operator Name**: Name of the operator
    2. **Explanation**: The explanation of the operator about what it does
    3. **Parameter List**: List of parameters with their types and descriptions. For example, `image` (ee.Image): The input image
    4. **Return Type**: The return type of the operator

    ### Output
    1. **Standard Code**: Define a function that uses the given operator and returns the result.
    The function name should be (Data Type+ operator name + Task). For example, `ee.Image.NormalizedDifference`->`imageNormalizedDifferenceTask`.
    2. **Test Cases in Configuration File**: Include multiple test cases, each with parameters, expected answer path, and output type.

    ### GEE objects in params
    1.If the parameter is an GEE object(e.g. ee.Image, ee.Number, etc), use the following format in the configuration file to return the object with python:
    param_name: !python |
        def get_ee_object():
            import ee
            ee.Initialize()
            # then get and return the wanted object
    2.Notice that some operators may require specific GEE objects as input. e.g. 'ee.Array.CholoskyDecomposition' requires a positive definite ee.Array matrix.
    
    ### Output Type
    1. The output type can be one of the following:
    GEE objects:
    "ee.Image", "ee.FeatureCollection", "ee.Number", "ee.List", "ee.Dictionary", "ee.Geometry", "ee.Array", "ee.ImageArray"
    Python objects:
    "str", "int", "float", "bool", "list", "dict", "NoneType"
    2. You can use other types if needed.
    
    ### Expected answer
    1. The value of the "expected_answer" field in the configuration file MUST be the path to the file containing the expected output.
    2. The file name should be (function name + "_testcase" + testcase_number), file type should be .npy for images and arrays,
     .geojson for geometry or feature objects, .txt for other types.
    
    ### Example
    #### Example Input
    - **Operator Name**: `normalizedDifference`
    - **Function Explanation**: Compute the normalized difference between the given two bands
    - **Parameter List**:
      - `image` (ee.Image): The input image
      - `band1` (str): The name of the first band
      - `band2` (str): The name of the second band
    - **Return Type**: `ee.Image`

    #### Example Output
    ##### Standard Code
    ```python
    def imageCannyEdgeDetectorTask(image: ee.Image, threshold: float, sigma: float = 1.0) -> ee.Image:
    """Applies the Canny edge detection algorithm to an image. """
        canny_edge = ee.Algorithms.CannyEdgeDetector(image, threshold, sigma)
        return canny_edge
    ```
    ##### Test Cases
    ```yaml
    imageCannyEdgeDetectorTask:
    - params:
        image: !python |
          def get_image():
            import ee
            ee.Initialize()
            dataset = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
                           .filterBounds(ee.Geometry.Point([120, 30]))
                           .filterDate('2024-01-01', '2024-12-31'))
            img = dataset.first()
            region = ee.Geometry.Rectangle([120.05, 30.05, 120.1, 30.1])
            clipped_img = img.clip(region)
            return clipped_img
        sigma: 1.5
        threshold: 0.3
      expected_answer: imageCannyEdgeDetectorTask_testcase1.npy
      out_type: ee.Image
    - params:
        image: !python |
          def get_image():
            import ee
            ee.Initialize()
            dataset = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
                           .filterBounds(ee.Geometry.Point([8, 48]))
                           .filter(ee.Filter.lte('CLOUD_COVER', 10))
                           .filterDate('2023-06-01', '2023-8-31'))
            img = dataset.first()
            region = ee.Geometry.Rectangle([8.5, 47.3, 8.55, 47.35])
            clipped_img = img.clip(region)
            return clipped_img
        sigma: 2.0
        threshold: 0.5
      expected_answer: imageCannyEdgeDetectorTask_testcase2.npy
      out_type: ee.Image
    ```
    
    ### Note
    1. The function should just include ONE operator and return the result. They are used for automatic testing.
    2. If the output is a GEE object, do NOT perform getInfo() function. Just return the object.
    3. Use the given operator for your answer, do NOT use other methods or operators to solve the task.
    4. Any import statements, initialization statements or example usages are NOT needed.
    5. Do NOT add any explanation.

    ### Operator Information
    Here is the operator information:
    
    ''')


CONSTRUCT_COMBINATION = (
    '''
    ## Task Description
    You need to generate test code and configuration file entries for given Google Earth Engine (GEE) Python API operators.
    Each operator will have two parts: the standard code and the test cases in the configuration file.

    ### Input
    **Operators**: Name of the operator list with order, indicating a regular usage in GEE for spatiotemporal analysis.

    ### Output
    1. **Standard Code**: Define a function that uses the given operators and returns the result.
    The function name should be clear. It should have a docstring that explains the function and its parameters.
    2. **Test Cases in Configuration File**: Include multiple test cases, each with parameters, expected answer path, and output type.

    ### GEE objects in params
    1.If the parameter is an GEE object(e.g. ee.Image, ee.Number, etc), use the following format in the configuration file to return the object with python:
    param_name: !python |
        def get_ee_object():
            import ee
            ee.Initialize()
            # then get and return the wanted object
    2.Notice that some operators may require specific GEE objects as input.

    ### Output Type
    1. The output type can be one of the following:
    GEE objects:
    "ee.Image", "ee.FeatureCollection", "ee.Number", "ee.List", "ee.Dictionary", "ee.Geometry", "ee.Array", "ee.ImageArray"
    Python objects:
    "str", "int", "float", "bool", "list", "dict"
    2. You can use other types if needed, but there MUST be ONE output.

    ### Expected answer
    1. The value of the "expected_answer" field in the configuration file MUST be the path to the file containing the expected output.
    2. The file name should be (function name + "_testcase" + testcase_number), file type should be .npy for images and arrays,
     .geojson for geometry or feature objects, .txt for other types.

    ### Example
    #### Example Input
    - **Operators**: ['addBands', 'select', 'map']

    #### Example Output
    ##### Standard Code
    ```python
    def image_collection_add_bands(image: ee.Image, srcBands: list, dstBands: list, collection: ee.ImageCollection) -> ee.ImageCollection:
    """
    Add specified bands from a source image to each image in an image collection, and return the modified image collection.
    Params:
        image (ee.Image): The source image used to add new bands.
        srcBands (list): A list of band names from the source image to be added to each image in the collection.
        dstBands (list): A list of new band names after adding the bands. Must have the same length as srcBands.
        collection (ee.ImageCollection): The image collection to be processed.
    Returns:
        ee.ImageCollection: A new image collection containing the selected bands.
    """
    def add_and_select(img):
        new_bands = image.select(srcBands)
        img_with_bands = img.addBands(new_bands, dstBands)
        return img_with_bands.select(img.bandNames())

    return collection.map(add_and_select)
    ```
    ##### Test Cases
    ```yaml
    image_collection_add_bands:
    - params:
        image: !python |
            def get_image():
                import ee
                ee.Initialize()
                return ee.Image.constant(1).rename('constant_band')
        srcBands: ['constant_band']
        dstBands: ['new_band']
        collection: !python |
            def get_image_collection():
                import ee
                ee.Initialize()
                return ee.ImageCollection([ee.Image.constant(2).rename('original_band')])
      expected_answer: image_collection_add_bands_testcase1.npy
      out_type: ee.ImageCollection
    - params:
        image: !python |
            def get_image():
                import ee
                ee.Initialize()
                dataset = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
                       .filterBounds(ee.Geometry.Point([120, 30]))
                       .filterDate('2024-01-01', '2024-12-31'))
                img = dataset.first()
                region = ee.Geometry.Rectangle([120.05, 30.05, 120.1, 30.1])
                clipped_img = img.clip(region)
                return clipped_img
        srcBands: ['SR_B4']
        dstBands: ['B11']
        collection: !python |
            def get_image_collection():
                import ee
                ee.Initialize()
                return = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
                       .filterBounds(ee.Geometry.Point([8, 48]))
      expected_answer: image_collection_add_bands_testcase2.npy
      out_type: ee.ImageCollection
    ```

    ### Note
    1. The tasks are used for automated testing, so they should be clear and concise.
    2. Use the given operators and order for the task, but you can add additional operators or delete some of them if needed.
    3. Do NOT use a whole Image to reduce computation, clip it into a smaller region first.
    4. If the output is a GEE object, do NOT perform getInfo() function. Just return the object.
    5. Any import statements or initialization statements are NOT needed.
    6. Do NOT add any explanation or example usages.
    
    ### Operators
    Here are the operators:

    ''')

CONSTRUCT_THEME = (
    '''
    ## Task Description
    You need to generate test code and configuration file entries for given Google Earth Engine (GEE) Python API analysis theme.
    The output will have two parts: the standard code and the test cases in the configuration file.

    ### Input
    **Theme**: The required theme of the task

    ### Output
    1. **Standard Code**: Define an analysis function with the theme and returns the result.
    The function name should be clear. It should have a docstring that explains the function and its parameters.
    2. **Test Cases in Configuration File**: Include multiple test cases, each with parameters, expected answer path, and output type.

    ### GEE objects in params
    1.If the parameter is an GEE object(e.g. ee.Image, ee.Number, etc), use the following format in the configuration file to return the object with python:
    param_name: !python |
        def get_ee_object():
            import ee
            ee.Initialize()
            # then get and return the wanted object
    2.Notice that some operators may require specific GEE objects as input.

    ### Output Type
    1. The output type can be one of the following:
    GEE objects:
    "ee.Image", "ee.FeatureCollection", "ee.Number", "ee.List", "ee.Dictionary", "ee.Geometry", "ee.Array", "ee.ImageArray"
    Python objects:
    "str", "int", "float", "bool", "list", "dict"
    2. You can use other types if needed, but there MUST be ONE output.

    ### Expected answer
    1. The value of the "expected_answer" field in the configuration file MUST be the path to the file containing the expected output.
    2. The file name should be (function name + "_testcase" + testcase_number), file type should be .npy for images and arrays,
     .geojson for geometry or feature objects, .txt for other types.

    ### Example
    ##### Standard Code
    ```python
    def image_collection_add_bands(image: ee.Image, srcBands: list, dstBands: list, collection: ee.ImageCollection) -> ee.ImageCollection:
    """
    Add specified bands from a source image to each image in an image collection, and return the modified image collection.
    Params:
        image (ee.Image): The source image used to add new bands.
        srcBands (list): A list of band names from the source image to be added to each image in the collection.
        dstBands (list): A list of new band names after adding the bands. Must have the same length as srcBands.
        collection (ee.ImageCollection): The image collection to be processed.
    Returns:
        ee.ImageCollection: A new image collection containing the selected bands.
    """
    def add_and_select(img):
        new_bands = image.select(srcBands)
        img_with_bands = img.addBands(new_bands, dstBands)
        return img_with_bands.select(img.bandNames())

    return collection.map(add_and_select)
    ```
    ##### Test Cases
    ```yaml
    image_collection_add_bands:
    - params:
        image: !python |
            def get_image():
                import ee
                ee.Initialize()
                return ee.Image.constant(1).rename('constant_band')
        srcBands: ['constant_band']
        dstBands: ['new_band']
        collection: !python |
            def get_image_collection():
                import ee
                ee.Initialize()
                return ee.ImageCollection([ee.Image.constant(2).rename('original_band')])
      expected_answer: image_collection_add_bands_testcase1.npy
      out_type: ee.ImageCollection
    - params:
        image: !python |
            def get_image():
                import ee
                ee.Initialize()
                dataset = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
                       .filterBounds(ee.Geometry.Point([120, 30]))
                       .filterDate('2024-01-01', '2024-12-31'))
                img = dataset.first()
                region = ee.Geometry.Rectangle([120.05, 30.05, 120.1, 30.1])
                clipped_img = img.clip(region)
                return clipped_img
        srcBands: ['SR_B4']
        dstBands: ['B11']
        collection: !python |
            def get_image_collection():
                import ee
                ee.Initialize()
                return = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
                       .filterBounds(ee.Geometry.Point([8, 48]))
      expected_answer: image_collection_add_bands_testcase2.npy
      out_type: ee.ImageCollection
    ```

    ### Note
    1. The tasks are used for automated testing, so they MUST be clear and concise.
    2. Do NOT use a whole Image to reduce computation, clip it into a smaller region first.
    3. If the output is a GEE object, do NOT perform getInfo() function. Just return the object.
    4. Any import statements or initialization statements are NOT needed in the standard code.
    5. Do NOT add any explanation or example usages.

    ### Theme
    Here is the theme:

    ''')

CONSTRUCT_THEME_NEW = (
    '''
    ## Task Description
    Generate test code and configuration file entries for a given Google Earth Engine (GEE) Python API analysis task.
    The tests are always based on the **GEE Python API**, whatever the platform or language the input is.
    Your output will have two parts: the standard code and test cases in YAML format.

    ### Input
    The input contains three parts:
    - **instruction**: The task description
    - **input**: Additional input information (may be empty)
    - **output**: The reference solution code

    ### Output
    1. **Standard Code**: Define a GEE analysis function that implements the core functionality from the reference solution.
       - Use a descriptive function name
       - Include proper type hints for inputs and outputs, give explanations for each parameter
       - Add clear and comprehensive docstring explaining the function's purpose and specifying the parameters and return type

    2. **Test Cases in YAML Format**: Include at least two test cases with:
       - Parameters
       - Expected answer path
       - Output type

    ### GEE Objects in Parameters
    For GEE objects (ee.Image, ee.FeatureCollection, etc.), use this format in the YAML:
    ```yaml
    param_name: !python |
      def get_ee_object():
        import ee
        ee.Initialize()
        # Code to create and return the GEE object
    ```

    ### Output Type
    Specify one of the following output types:
    - GEE objects: "ee.Image", "ee.FeatureCollection", "ee.Number", "ee.List", "ee.Dictionary", "ee.Geometry", "ee.Array", "ee.ImageCollection"
    - Python objects: "str", "int", "float", "bool", "list", "dict"

    ### Expected Answer
    - The "expected_answer" field must be a file path: (function_name + "_testcase" + testcase_number)
    - File extensions: .npy for images/arrays, .geojson for geometry/features, .txt for other types

    ### Important Guidelines
    1. Use small geographic regions to reduce computation (clip images)
    2. Return GEE objects directly (NO getInfo())
    3. Do NOT include import statements and initialization in the standard code
    4. Do NOT include explanations or example usage outside the required sections
    5. Test cases should use realistic GEE datasets when possible
    6. Ensure test cases cover different parameter configurations or regions
    7. Focus on the core spatiotemporal analysis, NO map visualization, the tasks should be complete and concise

    ### Example
    ##### Standard Code
    ```python
    def image_collection_add_bands(image: ee.Image, srcBands: list, dstBands: list, collection: ee.ImageCollection) -> ee.ImageCollection:
    """
    Add specified bands from a source image to each image in an image collection.

    Params:
        image (ee.Image): The source image used to add new bands.
        srcBands (list): A list of band names from the source image to be added.
        dstBands (list): A list of new band names after adding. Must match srcBands length.
        collection (ee.ImageCollection): The image collection to be processed.

    Returns:
        ee.ImageCollection: A new image collection with the added bands.
    """
    def add_and_select(img):
        new_bands = image.select(srcBands)
        img_with_bands = img.addBands(new_bands, dstBands)
        return img_with_bands.select(img.bandNames())

    return collection.map(add_and_select)
    ```
    ##### Test Cases
    ```yaml
    image_collection_add_bands:
    - params:
        image: !python |
          def get_image():
            import ee
            ee.Initialize()
            return ee.Image.constant(1).rename('constant_band')
        srcBands: ['constant_band']
        dstBands: ['new_band']
        collection: !python |
          def get_image_collection():
            import ee
            ee.Initialize()
            return ee.ImageCollection([ee.Image.constant(2).rename('original_band')])
      expected_answer: image_collection_add_bands_testcase1.npy
      out_type: ee.ImageCollection
    - params:
        image: !python |
          def get_image():
            import ee
            ee.Initialize()
            dataset = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
                     .filterBounds(ee.Geometry.Point([120, 30]))
                     .filterDate('2024-01-01', '2024-12-31'))
            img = dataset.first()
            region = ee.Geometry.Rectangle([120.05, 30.05, 120.1, 30.1])
            clipped_img = img.clip(region)
            return clipped_img
        srcBands: ['SR_B4']
        dstBands: ['B11']
        collection: !python |
          def get_image_collection():
            import ee
            ee.Initialize()
            return ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').filterBounds(ee.Geometry.Point([8, 48]))
      expected_answer: image_collection_add_bands_testcase2.npy
      out_type: ee.ImageCollection
    ```

    Based on the provided instruction, input, and output, create the standard code and test cases.
    If you think this task is impossible to complete in GEE Python API, please output "impossible".
    ''')

CONSTRUCT_THEME_2 = (
    '''
    ## Task Description
    Generate test code and configuration file entries for a given Google Earth Engine (GEE) Python API analysis task.
    The tests are always based on the **GEE Python API**, whatever the language the input is.
    Your output will have two parts: the standard code and test cases in YAML format.

    ### Input
    The input contains the reference to be referenced when designing the test code and test cases.

    ### Output
    1. **Standard Code**: Define a GEE analysis function that implements the core functionality from the reference solution.
       - Use a descriptive function name
       - Include proper type hints for inputs and outputs, give explanations for each parameter
       - Add clear and comprehensive docstring explaining the function's purpose and specifying the parameters and return type

    2. **Test Cases in YAML Format**: Include at least two test cases with:
       - Parameters
       - Expected answer path
       - Output type

    ### GEE Objects in Parameters
    For GEE objects (ee.Image, ee.FeatureCollection, etc.), use this format in the YAML:
    ```yaml
    param_name: !python |
      def get_ee_object():
        import ee
        ee.Initialize()
        # Code to create and return the GEE object
    ```

    ### Output Type
    Specify one of the following output types:
    - GEE objects: "ee.Image", "ee.FeatureCollection", "ee.Number", "ee.List", "ee.Dictionary", "ee.Geometry", "ee.Array", "ee.ImageCollection"
    - Python objects: "str", "int", "float", "bool", "list", "dict"

    ### Expected Answer
    - The "expected_answer" field must be a file path: (function_name + "_testcase" + testcase_number)
    - File extensions: .npy for images/arrays, .geojson for geometry/features, .txt for other types

    ### Important Guidelines
    1. Use small geographic regions to reduce computation (clip images)
    2. Return GEE objects directly (NO getInfo())
    3. Do NOT include import statements and initialization in the standard code
    4. Do NOT include explanations or example usage outside the required sections
    5. Test cases should use realistic GEE datasets when possible
    6. Ensure test cases cover different parameter configurations or regions
    7. Focus on the core spatiotemporal analysis, NO map visualization, the tasks should be complete and concise

    ### Example
    ##### Standard Code
    ```python
    def image_collection_add_bands(image: ee.Image, srcBands: list, dstBands: list, collection: ee.ImageCollection) -> ee.ImageCollection:
    """
    Add specified bands from a source image to each image in an image collection.

    Params:
        image (ee.Image): The source image used to add new bands.
        srcBands (list): A list of band names from the source image to be added.
        dstBands (list): A list of new band names after adding. Must match srcBands length.
        collection (ee.ImageCollection): The image collection to be processed.

    Returns:
        ee.ImageCollection: A new image collection with the added bands.
    """
    def add_and_select(img):
        new_bands = image.select(srcBands)
        img_with_bands = img.addBands(new_bands, dstBands)
        return img_with_bands.select(img.bandNames())

    return collection.map(add_and_select)
    ```
    ##### Test Cases
    ```yaml
    image_collection_add_bands:
    - params:
        image: !python |
          def get_image():
            import ee
            ee.Initialize()
            return ee.Image.constant(1).rename('constant_band')
        srcBands: ['constant_band']
        dstBands: ['new_band']
        collection: !python |
          def get_image_collection():
            import ee
            ee.Initialize()
            return ee.ImageCollection([ee.Image.constant(2).rename('original_band')])
      expected_answer: image_collection_add_bands_testcase1.npy
      out_type: ee.ImageCollection
    - params:
        image: !python |
          def get_image():
            import ee
            ee.Initialize()
            dataset = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
                     .filterBounds(ee.Geometry.Point([120, 30]))
                     .filterDate('2024-01-01', '2024-12-31'))
            img = dataset.first()
            region = ee.Geometry.Rectangle([120.05, 30.05, 120.1, 30.1])
            clipped_img = img.clip(region)
            return clipped_img
        srcBands: ['SR_B4']
        dstBands: ['B11']
        collection: !python |
          def get_image_collection():
            import ee
            ee.Initialize()
            return ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').filterBounds(ee.Geometry.Point([8, 48]))
      expected_answer: image_collection_add_bands_testcase2.npy
      out_type: ee.ImageCollection
    ```

    Based on the provided code, create the standard code and test cases. The code:
    ''')

CLARIFY_DOCSTRING = (
    '''
    I will provide you with a function that uses the GEE Python API.
    Your task is to:
    1. Understand what the function does by analyzing its code and logic.
    2. Add detailed English comments directly inside the docstring — without modifying the actual code.
    Ensure your explanation includes:
    1. A high-level summary of the function's purpose.
    2. Step-by-step breakdown of what it does.
    3. The setting of special parameters, including default values.
    4. Meaning of all input parameters and expected types.
    5. Return value description , including type and structure.
    NOTE:
    1. Make sure the final output is clear, well-structured.
    2. Add all the comments in the docstring, do NOT add them in the code body.
    3. Functions defined in the code body are NOT needed to be explained.
    4. Do NOT change the function body — only enhance the docstring and add explanatory comments where appropriate.
    5. NEVER mention the name operators are used in the function, just explain what the function and each step does.
    
    Directly provide the function with the enhanced docstring, without any additional text or explanation. 
    Warp the function with ```python  ```.
    ''')