package smos.exception;

import java.lang.Exception;

/**
  * This class represents the exception generated when attempting
  * to enter an entity without specifying a mandatory field
  */
public class MandatoryFieldException extends Exception {
	
	private static final long serialVersionUID = -4818814194670133466L;

	/**
	 * Generate the exception without an associated error message.
	 */
	public MandatoryFieldException() {
		super("Mandatory Field Missing!");
	}
	
	/**
	  * Generate the exception with an associated error message.
	  *
	  * @param pMessage The error message to be associated
	  * to the exception.
	  */
	public MandatoryFieldException(String pMessage) {
		super(pMessage);
	}
	
	
}