package smos.exception;

import java.lang.Exception;

/**
  * This class represents the exception generated when an entity
  * is not found in the database.
  */
public class EntityNotFoundException extends Exception {
	
	private static final long serialVersionUID = -1236105333523133721L;

	/**
	 * Generate the exception without an associated error message.
	 */
	public EntityNotFoundException() {
		super("Object Not Found in Repository!");
	}
	
	/**
	  * Generate the exception with an associated error message.
	  *
	  * @param pMessage 	The error message to be associated
      *                     to the exception.
	  */
	public EntityNotFoundException(String pMessage) {
		super(pMessage);
	}
	
	
}