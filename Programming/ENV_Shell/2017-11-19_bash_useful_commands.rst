*********************
Useful bash commands (macOS/Linux)
*********************

Find files including specific string recursively
------------------------------------------------

- Assume that using GNU grep, not the built-in version.
  
Find txt files in the current directory recursively
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

	.. code-block:: bash

		# grep
		grep -r --include "*.txt" .

	.. code-block:: bash
		
		# find w/ grep
		find . | grep "*.txt"

Find and remove txt files recursively
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

	.. code-block:: bash

		grep -lrZ --include "*.txt" | xargs rm -rf

		# -l: prints the entire file names
		# -r: searches recursively
		# -Z: prevents misinterpreting space character(s)
