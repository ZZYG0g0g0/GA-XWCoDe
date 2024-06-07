package smos.bean;

import smos.exception.InvalidValueException;

import java.io.Serializable;

/**
 * Class used to model a teaching.
 * 
 * @author Giulio D'Amora
 * @version 1.0
 * 
 *          2009 ?Copyright by SMOS
 * 
 */

public class Teaching implements Serializable {

	private static final long serialVersionUID = 2523612738702790957L;
	private int id_teaching;
	private String name;

	/**
	 * The constructor of the class.
	 */
	public Teaching() {
		this.id_teaching = 0;
	}

	/**
	 * The name of the course returns
	 * 
	 * @return The name of the course returns.
	 */
	public String getName() {
		return this.name;
	}

	/**
	 * Set the name of the course.
	 * 
	 * @param pName
	 *            Il nome da settare.
	 * 
	 * @throws InvalidValueException
	 */
	public void setName(String pName) throws InvalidValueException {
		if (pName.length() <= 4)// to verify the test
			throw new InvalidValueException();
		else
			this.name = pName;
	}

	/**
	 * Return the teaching id.
	 * 
	 * @return l'id dell'insegnamento.
	 */
	public int getId() {
		return this.id_teaching;
	}

	/**
	 * Set the teaching id.
	 * 
	 * @param pId
	 *            The id to be set..
	 */
	public void setId(int pId) {
		this.id_teaching = pId;
	}

}
