package smos.exception;
/**
 * This class represents the exception generated when a user
 * attempts to delete the only Admin user in the database.
 */

public class DeleteAdministratorException extends Exception {

	
	/**
	 * 
	 */
	private static final long serialVersionUID = -2081143475624381775L;

	/**
	 * Generate the exception without an associated error message.
	 * 
	 */
	public DeleteAdministratorException() {
		super("Unable to delete the user, the selected user is the only Admin present in the database! Create a new Manager and try again!");
	}
	
	/**
	  * Generate the exception with an associated error message.
	  *
	  * @param pMessage 	The error message to be associated
      *                     to the exception.
	  */
	public DeleteAdministratorException(String pMessage) {
		super(pMessage);
	}
}
