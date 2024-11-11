Here's a prioritized todo list for next steps, based on the OpenAI models integration and overall project structure:

## Core API Integration
- [ ] Fix the OpenAI API response handling - currently using dictionary access but should use object attributes
- [ ] Add error handling and retries for API calls
- [ ] Implement token counting for cost tracking
- [ ] Add typing for response objects and error cases

## Model Management
- [ ] Create a model registry/factory for different OpenAI models
- [ ] Add model-specific parameters (context windows, pricing, capabilities)
- [ ] Implement model validation and availability checking
- [ ] Add temperature and other parameter controls

## System Architecture  
- [ ] Set up proper logging system
- [ ] Create message history handler class/module
- [ ] Implement proper configuration management 
- [ ] Design clean interfaces between UI and model layers

## Testing
- [ ] Add unit tests for OpenAI API calls
- [ ] Create mock responses for testing
- [ ] Test error conditions and edge cases
- [ ] Add integration tests

## Interface Integration
- [ ] Create standard response format for UI consumption
- [ ] Add async support for non-blocking API calls
- [ ] Implement progress indicators/callbacks
- [ ] Design clean separation between model and UI layers

## Documentation
- [ ] Add detailed docstrings
- [ ] Create usage examples
- [ ] Document error handling approaches
- [ ] Add type hints throughout

The immediate focus should be on fixing the API response handling and adding proper error handling, as those are foundational for everything else.